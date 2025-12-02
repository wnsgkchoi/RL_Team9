# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
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
from agents.IDEFICS_agent import IDEFICSAgent
from agents.Phi2_agent import PHI2Agent
from environments.crafter_env import make_crafter_env

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

def instanciate_model(model_string, envs, num_prompt_images=1, use_text_description=False):
    if model_string == "CNN":
        return CNNAgent(envs)
    elif 'idefics' in model_string:
        return IDEFICSAgent(model_string, is_lora=True, padding_side="right", num_prompt_images=num_prompt_images, use_text_description=use_text_description)
    elif 'phi' in model_string:
        return PHI2Agent(model_string, is_lora=True, padding_side="right")
    else:
        raise Exception(f'the model you are trying to load {model_string} is not supported yet!')

# hydra loads the configuration from yaml file
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(args):

    #set tokenizer parallelization to false to avoid deadlocks
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    run_name = f"inference_{args.env_id}__{exp_name}__{args.seed}__{int(time.time())}"
    runs_directory = f"../runs_inference/{run_name}"

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

    # log only on the main process
    if accelerator.is_main_process:

        for key, value in args.items():
            accelerator.print(' ' + f'{key}: {value}')

        if args.track:
            accelerator.init_trackers(
                project_name=args.wandb_project_name,
                config=dict(args),
                init_kwargs={
                    "wandb": {
                        "entity": None,
                        "sync_tensorboard": True,
                        "name": run_name,
                        "monitor_gym": True,
                        "save_code": True,
                        "dir": args.wandb_log_dir
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

    # CRUCIAL: note that we needed to pass a different seed for each worker
    seed += int(args.local_rank) * 100
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # create vector env (you may use also AsyncVectorEnv)
    envs = gym.vector.AsyncVectorEnv(
        [make_crafter_env(runs_directory, 
                          save_stats=args.save_stats, 
                          save_episode=args.save_episode, 
                          save_video=args.save_video and i == 0 and accelerator.is_main_process,
                          save_video_every=args.save_video_every,
                          area=(args.env_area, args.env_area), 
                          view=(args.env_view, args.env_view), 
                          size=(args.env_size, args.env_size),
                          reward=True,
                          length=10000, 
                          seed=i+1+seed,
                          max_steps=2) for i in range(args.local_num_envs)],
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # create the agent
    agent = instanciate_model(args.model, envs, args.num_prompt_images, args.use_text_description)
    accelerator.print(f"the LORA adepted layers are: {agent.get_lora_module_names()}")

    # get back to different torch seeds per process after initializing the model.
    torch.manual_seed(seed)
    
    # adjust the learning rate as in: https://huggingface.co/docs/accelerate/concept_guides/performance
    args.learning_rate *= accelerator.num_processes
    args.critic_learning_rate *= accelerator.num_processes

    optimizer = optim.Adam([{'params': agent.network.parameters()}, \
                            {'params': agent.critic.parameters(), 'lr':args.critic_learning_rate}], lr=args.learning_rate, eps=args.adam_epsilon)

    # ask accelerate to set the devices properly for agent and optimizer
    agent, optimizer = accelerator.prepare(agent, optimizer)

    print('loading accelerator state')
    accelerator.load_state('<PATH_TO_CHECKPOINT>')
    print('loaded accelerator state')

    # check models init weights are the same (needed for random init and distributed setup)
    assert maybe_check_model_weigths_equal(agent, accelerator), f"model weigth initialization is different among the process instances"

    # the things that have not run through .prepare still need an explicit device placement
    device = accelerator.device

    # ALGO Logic: Storage setup
    text_obs = [] # bisogna che sia allineato con le obs!
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

    for iteration in tqdm(range(1, args.num_iterations + 1)):
        
        #set the agent in eval mode to collect trajectories
        if accelerator.num_processes > 1:
            agent.module.network.eval()
        else:
            agent.network.eval()


        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            text_obs.append(next_text_obs)
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                res = agent.module.get_action_and_value(next_obs, text_description=next_text_obs) if accelerator.num_processes > 1 else agent.get_action_and_value(next_obs, text_description=next_text_obs)
                action, logprob, value = res['action'], res['log_prob'], res['values']
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_text_obs = infos.get('obs', [None]*args.local_num_envs)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

        if accelerator.is_main_process:
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            live_metrics = read_stats_live(os.path.join(runs_directory, 'stats.jsonl'), run_name, 
               'ppo', verbose=False)
            if live_metrics:
                for k,v in live_metrics.items():
                    writer.add_scalar(k, v, global_step=global_step)

    envs.close()
    if accelerator.is_main_process:
        writer.close()
        if args.track:
            accelerator.end_training()

if __name__ == "__main__":
    train()

# run with NCCL_P2P_DISABLE=1 accelerate launch inference.py
# run with NCCL_P2P_DISABLE=1 accelerate launch --config_file <PATH_TO>/multi_gpu_accelerate_config.yaml inference.py
# run with NCCL_P2P_DISABLE=1 accelerate launch --config_file <PATH_TO>/multi_gpu_accelerate_config.yaml inference.py --config-name=idefics_config
# this launch it with the configuration set in the accelerate config file. you can find the default one at ~/.cache/huggingface/accelerate/default_config.yaml