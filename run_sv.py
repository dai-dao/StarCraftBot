import sys
import os
import shutil
import sys
import argparse

# Pytorch 
import torch
from torch.autograd import Variable
import torch.optim as optim
from rl.networks.pt_fully_conv import FullyConv
from rl.pre_processing import Preprocessor
from rl.supervised.parse import dataIter

# Workaround for pysc2 flags
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['run.py'])


parser = argparse.ArgumentParser(description='Starcraft 2 deep RL agents')
# Env specs
parser.add_argument('--res', type=int, default=64,
                    help='screen and minimap resolution')
# Network specs
parser.add_argument('--discount', type=float, default=0.95,
                    help='discount for future rewards')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed')
parser.add_argument('--gpu', type=str, default='0',
                    help='gpu device id')
parser.add_argument('--entropy_weight', type=float, default=1e-4,
                    help='weight of entropy penalty')
parser.add_argument('--max_gradient_norm', type=float, default=500.0,
                    help='Clip gradients')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                    help='Is using cuda or not')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
# Training
parser.add_argument('--data_dir', type=str, required=True,
                    help='data directory')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
# Bookeeping
args = parser.parse_args()


if not os.path.exists('out'):
    os.mkdir('out')
model_dir = os.path.join('out', 'pre_trained')
if not os.path.exists(model_dir):
    os.mkdir(model_dir)


def save_checkpoint(epoch_count, network, optimizer):
    print('Saving checkpoint at', os.path.join(model_dir, 'model.pth.tar'))
    out = {}
    params = network.get_trainable_params(with_id=True)
    for k, v in params.items():
        out[k] = v.state_dict()
    out['epoch']  = epoch_count
    out['optimizer'] = optimizer.state_dict()
    torch.save(out, os.path.join(model_dir, 'model.pth.tar'))
    

def main():
    # Define network
    network = FullyConv(args, supervised=True)
    if args.cuda:
        network.cuda()
    preproc = Preprocessor()
    # Optimizer and loss
    optimizer = optim.Adam(network.get_trainable_params())
    cross_ent = torch.nn.CrossEntropyLoss()
    # Process data files
    files = os.listdir(args.data_dir)
    unique = set([os.path.join(args.data_dir, f[:-5]) for f in files])
    pairs = [(u+'S.npz', u+'G.npz') for u in unique]
    # Training loop
    for index, pair in enumerate(pairs):
        for batch_ob, batch_action in dataIter(pairs[0], args):
            policy, _ = network.forward(batch_ob['screen'], batch_ob['minimap'], batch_ob['flat'])
            fn_id = policy[0]
            loss = cross_ent(fn_id, batch_action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if index % 500 == 0:
            print('At pair', index)
            print('Loss is', loss.cpu().data.numpy()[0])
            save_checkpoint(pair, network, optimizer)


if __name__ == "__main__":
    main()
