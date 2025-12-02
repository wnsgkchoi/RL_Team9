# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import json
import os
import random
import time
import hydra
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import wandb

from omegaconf import OmegaConf
from omegaconf import open_dict
from accelerate import Accelerator
from dataclasses import dataclass

from tqdm import tqdm

from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

from read_metrics import read_stats, read_stats_live

from agents.CNN_agent import CNNAgent
from agents.Idefics2_agent import Idefics2Agent
from agents.IDEFICS_agent import IDEFICSAgent
from agents.Phi2_agent import PHI2Agent
from agents.LLaMa2_agent import LLaMaAgent
from agents.LLaVa15_agent import LLaVa15Agent
from agents.Mistral_agent import MistralAgent
from agents.QWen_agent import QWenAgent
from environments.crafter_env import make_crafter_env
from environments.frozen_env import make_frozen_env
from environments.hanoi_env import make_hanoi_env
from environments.minigrid_env import make_minigrid_env
from PIL import Image

def evaluate_policy(agent, *, env, num_eval_episodes, accelerator):
    all_returns = []
    agent.eval()
    
    for _ in range(num_eval_episodes):
        ep_return = 0
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.predict(torch.tensor(obs, device=accelerator.device).unsqueeze(0))
            obs, reward, done, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
            done = done or truncated
            ep_return += reward
        all_returns.append(ep_return)
    return np.mean(all_returns), np.std(all_returns)


def augment_args_with_distributed_setup_variables(args):
    with open_dict(args):
        args.master_addr = os.environ.get('MASTER_ADDR')
        args.master_port = os.environ.get('MASTER_PORT')
        args.world_size = int(os.environ.get('WORLD_SIZE',"1"))
        args.rank = int(os.environ.get('RANK', "0"))
        args.local_rank = int(os.environ.get('LOCAL_RANK', "0"))

    return args

# Function to check if two model states are equal
def maybe_check_model_weigths_equal(agent, accelerator):
    if accelerator.num_processes == 1:
        return True
    accelerator.wait_for_everyone()
    state = agent.state_dict()
    for key in state.keys():
        # somehow accelerator.gather does not work with bitsandbytes__fp4 layers
        if "bitsandbytes" not in key:
            weight = accelerator.gather(state[key])
            dim_ = weight.shape[0] // accelerator.num_processes 
            for chunk in range(accelerator.num_processes -1):
                if torch.allclose(weight[chunk * dim_ :dim_ * (chunk + 1)], weight[dim_ * (chunk + 1): dim_ * (chunk + 2)]):
                    continue
                else:
                    return False
    return True

def instanciate_model(model_string, observation_space, action_enum, num_prompt_images=1, use_text_description=False, gradient_ckpt=False, lora=False, disable_adapters_for_generation=True):
    if model_string == "CNN":
        return CNNAgent(observation_space, len(action_enum))
    elif 'idefics2' in model_string:
        return Idefics2Agent(model_string,
                    action_enum=action_enum, 
                    is_lora=lora,
                    padding_side="right", 
                    num_prompt_images=num_prompt_images, 
                    use_text_description=use_text_description,
                    gradient_ckpt=gradient_ckpt,
                    disable_adapters_for_generation=disable_adapters_for_generation) 
    elif 'idefics' in model_string:
        return IDEFICSAgent(model_string,
                            action_enum=action_enum, 
                            is_lora=lora,
                            padding_side="right", 
                            num_prompt_images=num_prompt_images, 
                            use_text_description=use_text_description, 
                            gradient_ckpt=gradient_ckpt)
    elif 'phi' in model_string:
        return PHI2Agent(model_string,
                         action_enum=action_enum,
                         is_lora=lora, 
                         padding_side="right", 
                         gradient_ckpt=gradient_ckpt)
    elif 'Llama-2' in model_string:
        return LLaMaAgent(model_string,
                          action_enum=action_enum,
                         is_lora=lora, 
                         padding_side="right", 
                         gradient_ckpt=gradient_ckpt,
                         disable_adapters_for_generation=disable_adapters_for_generation)
    elif 'llava-1.5' in model_string:
        return LLaVa15Agent(model_string,
                            action_enum=action_enum, 
                            is_lora=lora,
                            padding_side="right", 
                            num_prompt_images=num_prompt_images, 
                            use_text_description=True, 
                            gradient_ckpt=gradient_ckpt)
    elif 'Mistral' in model_string:
        return MistralAgent(model_string,
                          action_enum=action_enum,
                         is_lora=lora, 
                         padding_side="right", 
                         gradient_ckpt=gradient_ckpt,
                         disable_adapters_for_generation=disable_adapters_for_generation)
    elif 'Qwen' in model_string:
        return QWenAgent(model_string,
            action_enum=action_enum, 
            is_lora=lora,
            padding_side="right", 
            num_prompt_images=num_prompt_images, 
            use_text_description=use_text_description, 
            gradient_ckpt=gradient_ckpt)
    else:
        raise Exception(f'the model you are trying to load {model_string} is not supported yet!')

def instanciate_envs(runs_directory, is_main_process, specifc_env_seed, **kwargs):
    # create vector env (you may use also AsyncVectorEnv)
    if kwargs['env_id'] == "CrafterReward-v1":
        envs = gym.vector.SyncVectorEnv(
            [make_crafter_env(runs_directory, 
                            save_stats=kwargs['save_stats'], 
                            save_episode=kwargs['save_episode'], 
                            save_video=kwargs['save_video'] and i == 0 and is_main_process,
                            save_video_every=kwargs['save_video_every'],
                            area=(kwargs['env_area'], kwargs['env_area']), 
                            view=(kwargs['env_view'], kwargs['env_view']), 
                            size=(kwargs['env_size'], kwargs['env_size']),
                            reward=True,
                            length=10000, 
                            seed=i+1+specifc_env_seed,
                            max_steps=1) for i in range(kwargs['local_num_envs'])],
        )
        eval_env = None
    elif kwargs['env_id'] in ["Hanoi3Disk-v0", "Hanoi4Disk-v0"]:
        envs = gym.vector.SyncVectorEnv(
            [
                make_hanoi_env(
                    runs_directory,
                    env_name=kwargs['env_id'],
                    save_stats=kwargs['save_stats']
                )
                for i in range(kwargs['local_num_envs'])
            ]
        )
        eval_env = None
    elif "MiniGrid" in kwargs['env_id'] or "BabyAI" in kwargs['env_id']:
        envs = gym.vector.SyncVectorEnv(
            [
                make_minigrid_env(
                    runs_directory,
                    kwargs['env_id'], #'MiniGrid-LavaCrossingS11N5-v0'
                    possible_actions_list=kwargs['possible_actions_list'],
                    fov=kwargs['fov'],
                    fixed_orientation=kwargs['fixed_orientation'],
                    no_step_description=kwargs['no_step_description'],
                    seed=specifc_env_seed + i * 100, #this seeding ensures enought seed distance from one env to the other to allow for multireset (ReseedWrapper)
                    save_video=kwargs['save_video'] and i == 0 and is_main_process,
                    save_video_every=kwargs['save_video_every'],
                    save_stats=kwargs['save_stats'],
                    reseed_env=kwargs['reseed_env'],
                    first_person=kwargs['first_person'])
                for i in range(kwargs['local_num_envs'])
            ]
        )
        eval_env = None
    elif kwargs['env_id'] == "FrozenLakeText-v0":
        envs = gym.vector.SyncVectorEnv(
            [make_frozen_env(
                runs_directory,
                area=kwargs['env_area'], # 8x8,
                fov=kwargs['fov'],
                seed=specifc_env_seed + i + 1,
                size=(kwargs['env_size'], kwargs['env_size']),
                is_slippery=kwargs['is_slippery'],
                fixed_orientation=kwargs['fixed_orientation'],
                save_video=kwargs['save_video'] and i == 0 and is_main_process,
                save_video_every=kwargs['save_video_every'],
                save_stats=kwargs['save_stats'],
                first_person=kwargs['first_person']
            )
            for i in range(kwargs['local_num_envs'])
            ]
        )
        eval_env = make_frozen_env('',
                                area=kwargs['env_area'],
                                fov=kwargs['fov'],
                                size=(kwargs['env_size'], kwargs['env_size']),
                                is_slippery=kwargs['is_slippery'],
                                fixed_orientation=kwargs['fixed_orientation'],
                                seed=kwargs['seed_eval'],
                                save_video=False,
                                save_stats=False)()
    else:
        raise Exception(f'Environment {kwargs["env_id"]} is not supported')
    return envs, eval_env

def save_initial_env_states(envs, save_dir, process_index=0):
    """Save initial environment states as images"""
    os.makedirs(save_dir, exist_ok=True)
    
    # For SyncVectorEnv, we need to access individual environments
    for env_idx in range(envs.num_envs):
        try:
            # Get the individual environment
            single_env = envs.envs[env_idx]
            
            # Unwrap to get to the actual environment that has render
            current_env = single_env
            while hasattr(current_env, 'env'):
                current_env = current_env.env
            
            # Render the environment
            if hasattr(current_env, 'render'):
                img_array = current_env.render()
                
                # Convert to PIL Image and save
                if img_array is not None:
                    if isinstance(img_array, np.ndarray):
                        img = Image.fromarray(img_array.astype(np.uint8))
                    else:
                        img = img_array
                    
                    save_path = os.path.join(save_dir, f"process_{process_index}_env_{env_idx}_initial.png")
                    img.save(save_path)
                    print(f"Saved initial environment state to {save_path}")
        except Exception as e:
            print(f"Could not save environment {env_idx}: {e}")

# hydra loads the configuration from yaml file
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(args):
    if args.suppress_warnings:
        import warnings
        warnings.filterwarnings("ignore")

    #set tokenizer parallelization to false to avoid deadlocks
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    model_name_last_two_parts = '-'.join(args.model.split('/')[-2:])
    run_name = ('INFERENCE___' if args.disable_training else '') + f"{args.env_id}__{model_name_last_two_parts}__{exp_name}__{args.seed}__{str(args.wandb_log_dir)}"
    runs_directory = f"../runs/{run_name}"

    os.makedirs(runs_directory, exist_ok=True)

    # Initialize Accelerator
    log_with = 'wandb' if args.track else None
    accelerator = Accelerator(project_dir=runs_directory, log_with=log_with)
    args = augment_args_with_distributed_setup_variables(args)

    # compute and add some variables for distributed training
    with open_dict(args):
        args.local_batch_size = int(args.local_num_envs) * int(args.num_steps)
        args.local_minibatch_size = int(args.local_batch_size) // int(args.num_minibatches)
        args.num_envs = int(args.local_num_envs) * int(args.world_size)

        args.batch_size = int(args.num_envs) * int(args.num_steps)
        args.minibatch_size = int(args.batch_size) // int(args.num_minibatches)
        args.num_iterations = int(args.total_timesteps) // int(args.batch_size)
        args.gradient_accumulation = int(args.gradient_accumulation)

        args.eval_interval = int(args.eval_interval)
        args.num_eval_episodes = int(args.num_eval_episodes)

        args.save_video_every = int(args.save_video_every)
        args.possible_actions_list = args.possible_actions_list.split()

    # log only on the main process
    if accelerator.is_main_process:

        accelerator.print(f"[INFO]: {runs_directory=}")

        for key, value in args.items():
            accelerator.print(' ' + f'{key}: {value}')

        if args.track:
            accelerator.init_trackers(
                project_name=args.wandb_project_name,
                config=dict(args),
                init_kwargs={
                    "wandb": {
                        "entity": args.wandb_entity,
                        "sync_tensorboard": True,
                        "name": run_name,
                        "monitor_gym": True,
                        "save_code": True,
                        "dir": runs_directory
                        }}
                        )
        writer = SummaryWriter(runs_directory)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    # TRY NOT TO MODIFY: seeding
    seed = int(args.seed)
    # if training a RANDOM network we need the same initialization among workers
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CRUCIAL: note that we needed to pass a different seed for each worker
    seed += int(args.local_rank) * 100
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # ENVIRONMENTS creation
    envs, eval_env = instanciate_envs(runs_directory, accelerator.is_main_process, seed, **args)
    # dummy variable used later to flag for specific crafter logging
    is_crafter = True if args.env_id == "CrafterReward-v1" else False
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # get the obs dimensions, useful to compute the CNN agent kernels
    observation_space = envs.single_observation_space
    print(f"{observation_space=}")
    agent = instanciate_model(args.model, observation_space, envs.envs[0].action_enum, args.num_prompt_images, args.use_text_description, args.gradient_ckpt, args.lora, args.disable_adapters_for_generation)

    if accelerator.is_main_process:
        if args.track:
            columns = ["value prompt", "generated action"] + [action.value for action in envs.envs[0].action_enum]
            action_generation_table = wandb.Table(columns=columns)

    # get back to different torch seeds per process after initializing the model.
    torch.manual_seed(seed)

    # adjust the learning rate as in: https://huggingface.co/docs/accelerate/concept_guides/performance
    #args.learning_rate *= accelerator.num_processes
    #args.critic_learning_rate *= accelerator.num_processes

    optimizer = optim.Adam([{'params': agent.network.parameters()}, \
                            {'params': agent.critic.parameters(), 'lr':args.critic_learning_rate}], lr=args.learning_rate, eps=args.adam_epsilon)

    # ask accelerate to set the devices properly for agent and optimizer
    agent, optimizer = accelerator.prepare(agent, optimizer)

    # if a ckpt is provided, load it
    if args.model_ckpt:
        accelerator.load_state(args.model_ckpt)
        accelerator.print(f'loaded accelerator state form {args.model_ckpt}')

    if args.from_accelerate_savestate_to_checkpoint and args.model_ckpt:
        accelerator.print(f'saving {args.model_ckpt} to pytorch checkpoint (loadable by from_pretrained())')
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            output_dir = os.path.join(f"{args.model_ckpt}_pytorch_ckpt")
            accelerator.print(f'saving to {output_dir}.')
            unwrapped_model = accelerator.unwrap_model(agent.module)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)

    if accelerator.num_processes > 1:
        agent = agent.module

    accelerator.print(f"the LORA adepted layers are: {agent.get_lora_module_names()}")
    accelerator.print(f"the agent is: {agent}")

    # check models init weights are the same (needed for random init and distributed setup)
    assert maybe_check_model_weigths_equal(agent, accelerator), f"model weigth initialization is different among the process instances"

    # the things that have not run through .prepare still need an explicit device placement
    device = accelerator.device

    # ALGO Logic: Storage setup
    text_obs = [] # bisogna che sia allineato con le obs!
    #print(f"{args.num_steps=}, {args.local_num_envs=}, {envs.single_observation_space.shape=}")
    obs = torch.zeros((args.num_steps, args.local_num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.local_num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.local_num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.local_num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.local_num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.local_num_envs)).to(device)


    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, infos = envs.reset(seed=seed)  #infos is empty after reset
    next_text_obs = infos.get('obs', [None]*args.local_num_envs)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.local_num_envs).to(device)
    
    # Save initial environment states
    if accelerator.is_main_process:
        initial_env_save_dir = os.path.join(runs_directory, 'initial_env_states')
        save_initial_env_states(envs, initial_env_save_dir, process_index=accelerator.process_index)

    for iteration in tqdm(range(1, args.num_iterations + 1)):
        
        #set the agent in eval mode to collect trajectories
        agent.network.eval()
        agent.critic.eval()

        #debug file
        if args.debug:
            accelerator.print("[INFO] creating debugging files with run logs")
            debug_path = os.path.join(runs_directory, 'debug_logs')
            os.makedirs(debug_path, exist_ok=True)
            global file_debug
            file_debug = open(os.path.join(debug_path, f'debug_itaration_{accelerator.process_index}_{iteration}.txt'), 'w')
            jsonify_infos = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in infos.items()}
            file_debug.write(f"RESET: \n{next_text_obs=}, {next_obs.shape=}\n")
            file_debug.write(json.dumps(jsonify_infos, indent=4))

        total_action_match_found_in_iteration = 0

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            text_obs.append(next_text_obs)
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                res = agent.get_action_and_value(next_obs, 
                                                 value_prompt_template=args.value_prompt_template,
                                                 action_template=args.action_template,
                                                 text_description=next_text_obs, 
                                                 temperature=args.temperature, 
                                                 generate_actions=args.generate_actions,
                                                 normalization_by_words=args.normalization_by_words,
                                                 action_logits_from_whole_seq=args.action_logits_from_whole_seq,
                                                 advanced_action_matching=args.advanced_action_matching)
                action, logprob, value = res['action'], res['log_prob'], res['values']
                values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

            total_action_match_found_in_iteration += res.get('action_match_found', 0)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_text_obs = infos.get('obs', [None]*args.local_num_envs)
            
            if args.debug:
                if 'final_observation' in infos: #do not log all the image array
                    del infos['final_observation']
                jsonify_infos = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in infos.items()}
                file_debug.write(f"\n\nSTEP {step}:\nprompts - {res.get('prompts', 'no prompts')}")
                file_debug.write(f"\nAction {action}, {res['action_logits']}.\nresulted in:\n{next_obs.shape=}, {reward=}, {terminations=}, {truncations=}, \n{next_text_obs=}\n")
                file_debug.write(json.dumps(jsonify_infos, indent=4))


            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
        
        # save ckpt
        if iteration % args.save_every == 0:
            ckpt_name = f"agent_ckpt_iter-{iteration}_gs-{global_step}"
            accelerator.print(f"saving {ckpt_name} which corresponds to {global_step=}")
            # ask accelerator to save the ckpt. it will synch with the other processes
            accelerator.save_state(output_dir=os.path.join(runs_directory, ckpt_name))

        if not args.disable_training:  # if disable_training is true, we do not train the model 
            ### TRAINING 
            
            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs, 
                                            value_prompt_template=args.value_prompt_template,
                                            text_description=next_text_obs)['values'].reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_text_obs = [item for sublist in text_obs for item in sublist]
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.local_batch_size)
            clipfracs = []

            gradient_accumulation_counter = 0
            assert (args.local_batch_size % args.gradient_accumulation == 0) and (args.gradient_accumulation <= args.local_batch_size), f"{args.gradient_accumulation=} must perfectly divide the {args.local_batch_size=} and be smaller or equal"
            
            #set the agent in train mode to run ppo
            agent.network.train()
            agent.critic.train()

            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.local_batch_size, args.local_minibatch_size):
                    end = start + args.local_minibatch_size
                    mb_inds = b_inds[start:end]

                    res = agent.get_action_and_value(b_obs[mb_inds],
                                                    value_prompt_template=args.value_prompt_template,
                                                    action_template=args.action_template,
                                                    action=b_actions.long()[mb_inds], 
                                                    text_description=[b_text_obs[i] for i in mb_inds], 
                                                    temperature=args.temperature, 
                                                    generate_actions=args.generate_actions,
                                                    normalization_by_words=args.normalization_by_words,
                                                    advanced_action_matching=args.advanced_action_matching)
                    newlogprob, entropy, newvalue = res['log_prob'], res['entropy'], res['values']

                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    # if args.gradient_accumulation = 1 we have normal flow
                    loss = loss / args.gradient_accumulation
                    accelerator.backward(loss)

                    # if time to update, do it.
                    # Note: with this code there is probably a slow down due to oversync of gradients: https://huggingface.co/docs/accelerate/concept_guides/gradient_synchronization
                    if (gradient_accumulation_counter+1) % args.gradient_accumulation == 0:
                        #sync grads
                        if args.world_size > 1:
                            # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
                            all_grads_list = []
                            for param in agent.parameters():
                                if param.grad is not None:
                                    all_grads_list.append(param.grad.view(-1))
                            all_grads = torch.cat(all_grads_list)
                            dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
                            offset = 0
                            for param in agent.parameters():
                                if param.grad is not None:
                                    param.grad.data.copy_(
                                        all_grads[offset : offset + param.numel()].view_as(param.grad.data) / args.world_size
                                    )
                                    offset += param.numel()

                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                    gradient_accumulation_counter += 1

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break
            
            text_obs = []

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            ### END TRAINING 
        
        if args.debug:
            file_debug.close()

        if accelerator.is_main_process:
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if not args.disable_training:
                writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
                writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar("charts/total_action_match_found_in_iteration", total_action_match_found_in_iteration, global_step)
            writer.add_scalar("charts/action_generation_cache_dimension", len(agent.action_generation_cache), global_step)

            live_metrics = read_stats_live(os.path.join(runs_directory, 'stats.jsonl'), run_name, 
               'ppo', verbose=False, is_crafter=is_crafter)
            if live_metrics:
                for k,v in live_metrics.items():
                    writer.add_scalar(k, v, global_step=global_step)

            if args.track:
                for k,v in agent.action_generation_cache.items():
                    action_generation_table.add_data(k, v[1], *v[0]) 
                accelerator.log({"action table": action_generation_table}, step=global_step)

            if iteration % args.eval_interval == 0:
                accelerator.print(f"Evaluating the model at global step: {global_step=}")
                if eval_env is not None:
                    avg_return, _ = evaluate_policy(agent, env=eval_env, num_eval_episodes=args.num_eval_episodes, accelerator=accelerator)
                    writer.add_scalar("charts/eval_avg_return", avg_return, global_step)
                    accelerator.print(f"Evaluation finished at global step: {global_step}, {avg_return=}")
                else:
                    print(f'[Warning]: no evaluation environment specified for env {args.env_id}. Skipping evaluation...')

    # save last ckpt
    ckpt_name = f"agent_ckpt_last_gs-{global_step}"
    accelerator.print(f"saving {ckpt_name} which corresponds to {global_step=}")
    accelerator.save_state(output_dir=os.path.join(runs_directory, ckpt_name))

    # compute metrics
    if accelerator.is_main_process:
        read_stats(os.path.join(os.getcwd(), runs_directory),
               runs_directory, 
               run_name, 
               'ppo', 
               budget=args.total_timesteps,
               is_crafter=is_crafter)

    envs.close()
    if accelerator.is_main_process:
        writer.close()
        if args.track:
            accelerator.end_training()

if __name__ == "__main__":
    train()

# run with NCCL_P2P_DISABLE=1 accelerate launch ppo_crafter_parallel.py
# run with NCCL_P2P_DISABLE=1 accelerate launch --config_file <PATH_TO>/multi_gpu_accelerate_config.yaml ppo_crafter_parallel.py
# run with NCCL_P2P_DISABLE=1 accelerate launch --config_file <PATH_TO>/multi_gpu_accelerate_config.yaml ppo_crafter_parallel.py --config-name=idefics_config
# this launch it with the configuration set in the accelerate config file. you can find the default one at ~/.cache/huggingface/accelerate/default_config.yaml