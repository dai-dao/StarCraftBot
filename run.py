import sys
import os
import shutil
import sys
import argparse
from functools import partial

from logger import Logger
from rl.agents.a2c.runner import A2CRunner
from rl.environment import SubprocVecEnv, make_sc2env, SingleEnv

# Pytorch 
import torch
from rl.agents.a2c.pt_agent import A2CAgent


# Workaround for pysc2 flags
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['run.py'])


parser = argparse.ArgumentParser(description='Starcraft 2 deep RL agents')
parser.add_argument('--experiment_id', type=str, required=True, 
                    help='identifier to store experiment results')
parser.add_argument('--eval', action='store_true',
                    help='if false, episode scores are evaluated')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='overwrite existing experiments (if --train=True)')
parser.add_argument('--map', type=str, default='MoveToBeacon',
                    help='name of SC2 map')
parser.add_argument('--vis', action='store_true',
                    help='render with pygame')
parser.add_argument('--max_windows', type=int, default=1,
                    help='maximum number of visualization windows to open')
parser.add_argument('--res', type=int, default=32,
                    help='screen and minimap resolution')
parser.add_argument('--envs', type=int, default=2,
                    help='number of environments simulated in parallel')
parser.add_argument('--step_mul', type=int, default=8,
                    help='number of game steps per agent step')
parser.add_argument('--steps_per_batch', type=int, default=16,
                    help='number of agent steps when collecting trajectories for a single batch')
parser.add_argument('--discount', type=float, default=0.95,
                    help='discount for future rewards')
parser.add_argument('--iters', type=int, default=-1,
                    help='number of iterations to run (-1 to run forever)')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed')
parser.add_argument('--gpu', type=str, default='0',
                    help='gpu device id')
parser.add_argument('--summary_iters', type=int, default=1,
                    help='record summary after this many iterations')
parser.add_argument('--save_iters', type=int, default=500,
                    help='store checkpoint after this many iterations')
parser.add_argument('--max_to_keep', type=int, default=5,
                    help='maximum number of checkpoints to keep before discarding older ones')
parser.add_argument('--entropy_weight', type=float, default=1e-4,
                    help='weight of entropy penalty')
parser.add_argument('--value_loss_weight', type=float, default=1.0,
                    help='weight of value function loss')
parser.add_argument('--max_gradient_norm', type=float, default=500.0,
                    help='Clip gradients')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                    help='Is using cuda or not')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--save_dir', type=str, default='out/models',
                    help='root directory for checkpoint storage')
parser.add_argument('--summary_dir', type=str, default='out/summary',
                    help='root directory for summary storage')

args = parser.parse_args()
# TODO write args to config file and store together with summaries (https://pypi.python.org/pypi/ConfigArgParse)
args.train = not args.eval
ckpt_path = os.path.join(args.save_dir, args.experiment_id)
summary_type = 'train' if args.train else 'eval'
summary_path = os.path.join(args.summary_dir, args.experiment_id, summary_type)

args.save_dir = ckpt_path
args.summary_dir = summary_path


def main():
    size_px = (args.res, args.res)
    env_args = dict(
        map_name=args.map,
        step_mul=args.step_mul,
        game_steps_per_episode=0,
        screen_size_px=size_px,
        minimap_size_px=size_px)

    vis_env_args = env_args.copy()
    vis_env_args['visualize'] = args.vis
    
    num_vis = min(args.envs, args.max_windows)
    env_fns = [partial(make_sc2env, **vis_env_args)] * num_vis
    num_no_vis = args.envs - num_vis
    if num_no_vis > 0:
      env_fns.extend([partial(make_sc2env, **env_args)] * num_no_vis)
    envs = SubprocVecEnv(env_fns)
    agent = A2CAgent(args)

    current_epoch = 0
    if os.path.isfile(args.save_dir + '.pth.tar') and not args.overwrite:
      current_epoch = agent.load_checkpoint()
      print("Restored from last checkpoint at epoch", current_epoch)

    summary_writer = Logger(args.summary_dir)
    runner = A2CRunner(
        envs=envs,
        agent=agent,
        train=args.train,
        summary_writer=summary_writer,
        discount=args.discount,
        n_steps=args.steps_per_batch)

    runner.reset()
    try:
      while True:
        if current_epoch % args.save_iters == 0:
           agent.save_checkpoint(current_epoch) 
        result = runner.run_batch(train_summary=True)
        # agent.log(summary_writer, i)
        current_epoch += 1
    except KeyboardInterrupt:
        pass

    envs.close()
    print('mean score: %f' % runner.get_mean_score())

if __name__ == "__main__":
    main()
