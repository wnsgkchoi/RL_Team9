import collections
import json
import pathlib
import numpy as np
import common

def get_frozen_lake_stats(filename):
  rewards = []
  lengths = []
  success_count = 0
  metrics = {}
  for line in filename.read_text().split('\n'):
    if not line.strip():
      continue
    episode = json.loads(line)
    lengths.append(episode['length'])
    rewards.append(episode['reward'])
    if episode['reward'] >= 0.3:  # Assuming 3.0 is the reward for success
        success_count += 1
  
  metrics['Length'] = np.mean(np.array(lengths))
  metrics['Reward'] = np.mean(np.array(rewards))
  metrics['Cumulative_Reward'] = np.sum(np.array(rewards))
  metrics['Success_Count'] = success_count
  metrics['Failed_Count'] = len(rewards) - success_count
  metrics['Success_Rate'] = success_count / len(rewards) if len(rewards) > 0 else 0.0
  return metrics

def read_stats_live(file_path, task, method, verbose=False, is_crafter=True):
  metrics = {}
  runs = []

  if not is_crafter:
    metrics = get_frozen_lake_stats(pathlib.Path(file_path))
    return metrics

  rewards, lengths, achievements = load_stats(pathlib.Path(file_path), budget=float('inf'))
  if len(rewards) == 0:
    return None
  runs.append(dict(
      task=task,
      method=method,
      seed='0',
      xs=np.cumsum(lengths).tolist(),
      reward=rewards,
      length=lengths,
      **achievements,
  ))
  episodes = np.array([len(x['length']) for x in runs])
  rewards = np.array([np.mean(x['reward']) for x in runs])
  lengths = np.array([np.mean(x['length']) for x in runs])
  # the budget is the whole set of episodes
  budget = np.array([x['length'] for x in runs]).sum()
  percents, methods, seeds, tasks = common.compute_success_rates(
      runs, budget, sortby=0)
  scores = np.squeeze(common.compute_scores(percents))
  metrics['Score'] = np.mean(scores)
  metrics['Reward'] = np.mean(rewards)
  metrics['Length'] = np.mean(lengths)
  metrics['Episodes'] = np.mean(episodes)

  if verbose:
    for task, percent in sorted(tasks, np.squeeze(percents).T):
      name = task[len('achievement_'):].replace('_', '/').title()
      metrics[name] = np.mean(percent)
  return metrics

def read_stats(indir, outdir, task, method, budget=int(1e6), verbose=False, is_crafter=True):

  indir = pathlib.Path(indir)
  outdir = pathlib.Path(outdir)
  runs = []

  #simple hot fix for frozen lake env
  if not is_crafter:
    filename = indir / 'stats.jsonl'
    if not filename.exists():
        print(f"Stats file not found in {indir}")
        return
    metrics = get_frozen_lake_stats(pathlib.Path(filename))
    print(metrics)
    return
  
  print(f'Loading {indir.name}...')
  filenames = sorted(list(indir.glob('**/stats.jsonl')))
  for index, filename in enumerate(filenames):
    if not filename.is_file():
      continue
    rewards, lengths, achievements = load_stats(filename, budget)
    if sum(lengths) < budget - 1e4:
      message = f'Skipping incomplete run ({sum(lengths)} < {budget} steps): '
      message += f'{filename.relative_to(indir.parent)}'
      print(f'==> {message}')
      continue
    runs.append(dict(
        task=task,
        method=method,
        seed=str(index),
        xs=np.cumsum(lengths).tolist(),
        reward=rewards,
        length=lengths,
        **achievements,
    ))
  if not runs:
    print('No completed runs.\n')
    return
  print_summary(runs, budget, verbose)
  outdir.mkdir(exist_ok=True, parents=True)
  filename = (outdir / f'{task}-{method}.json')
  filename.write_text(json.dumps(runs))
  print('Wrote', filename)
  print('')
  return


def load_stats(filename, budget):
  steps = 0
  rewards = []
  lengths = []
  achievements = collections.defaultdict(list)
  for line in filename.read_text().split('\n'):
    if not line.strip():
      continue
    episode = json.loads(line)
    steps += episode['length']
    if steps > budget:
      break
    lengths.append(episode['length'])
    for key, value in episode.items():
      if key.startswith('achievement_'):
        achievements[key].append(value)
    unlocks = int(np.sum([(v[-1] >= 1) for v in achievements.values()]))
    health = -0.9
    rewards.append(unlocks + health)
  return rewards, lengths, achievements


def print_summary(runs, budget, verbose):
  episodes = np.array([len(x['length']) for x in runs])
  rewards = np.array([np.mean(x['reward']) for x in runs])
  lengths = np.array([np.mean(x['length']) for x in runs])
  percents, methods, seeds, tasks = common.compute_success_rates(
      runs, budget, sortby=0)
  scores = np.squeeze(common.compute_scores(percents))
  print(f'Score:        {np.mean(scores):10.2f} ± {np.std(scores):.2f}')
  print(f'Reward:       {np.mean(rewards):10.2f} ± {np.std(rewards):.2f}')
  print(f'Length:       {np.mean(lengths):10.2f} ± {np.std(lengths):.2f}')
  print(f'Episodes:     {np.mean(episodes):10.2f} ± {np.std(episodes):.2f}')
  if verbose:
    for task, percent in sorted(tasks, np.squeeze(percents).T):
      name = task[len('achievement_'):].replace('_', ' ').title()
      print(f'{name:<20}  {np.mean(percent):6.2f}%')
  return

if __name__ == "__main__":
  read_stats(
      '<PATH_TO_RUN>',
      'scores', 'crafter_reward', 'ppo', budget=25000)
