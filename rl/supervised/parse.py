import os 
import numpy as np
from scipy import sparse

import torch
from torch.autograd import Variable


def renormalize_screen(screen_features):
    screen_features[:, 0, :, :] = screen_features[:, 0, :, :] * 256
    screen_features[:, 1, :, :] = screen_features[:, 1, :, :] * 4
    screen_features[:, 2, :, :] = screen_features[:, 2, :, :] * 2
    screen_features[:, 3, :, :] = screen_features[:, 3, :, :] * 2
    screen_features[:, 4, :, :] = screen_features[:, 4, :, :] * 5
    screen_features[:, 5, :, :] = screen_features[:, 5, :, :] * 1850
    screen_features[:, 6, :, :] = screen_features[:, 6, :, :] * 16
    screen_features[:, 7, :, :] = screen_features[:, 7, :, :] * 256
    

def renormalize_minimap(minimap_features):
    minimap_features[:, 0, :, :] = minimap_features[:, 0, :, :] * 256  
    minimap_features[:, 1, :, :] = minimap_features[:, 1, :, :] * 4  
    minimap_features[:, 2, :, :] = minimap_features[:, 2, :, :] * 2  
    minimap_features[:, 3, :, :] = minimap_features[:, 3, :, :] * 5  


def renormalize_feats(feats):
    feats[:, 0] = 0 # Player id
    feats[:, 1] = feats[:, 1] * 66557
    feats[:, 2] = feats[:, 2] * 39180
    feats[:, 3] = feats[:, 3] * 200
    feats[:, 4] = feats[:, 4] * 200
    feats[:, 5] = feats[:, 5] * 200
    feats[:, 6] = feats[:, 6] * 200
    feats[:, 7] = feats[:, 7] * 122
    feats[:, 8] = feats[:, 8] * 166
    feats[:, 9] = feats[:, 9] * 0
    feats[:, 10] = feats[:, 10] * 19


def dataIter(pair, args):
    # S is spatial features
    S = np.asarray(sparse.load_npz(pair[0]).todense()).reshape([-1, 13, 64, 64])
    screen_features = S[:, 0:8, :, :]
    renormalize_screen(screen_features)
    minimap_features = S[:, 8:12, :, :]
    renormalize_minimap(minimap_features)
    # G is global features
    G = np.asarray(sparse.load_npz(pair[1]).todense())
    feats = G[:, 0:11]
    renormalize_feats(feats)
    # Actions
    actions_taken = G[:, 25]
    # Data loop
    n = screen_features.shape[0]
    cursor = 0


    while cursor < n:
        batch = min(n - cursor, args.batch_size)
        out = {}
        out['screen'] = Variable(torch.from_numpy(screen_features[cursor : cursor + batch]))
        out['minimap'] = Variable(torch.from_numpy(minimap_features[cursor : cursor + batch]))
        out['flat'] = Variable(torch.from_numpy(feats[cursor : cursor + batch]))
        action_var = Variable(torch.from_numpy(actions_taken[cursor : cursor + batch]).long())
        if args.cuda:
            action_var.cuda()
            for k, v in out.items():
                v.cuda()
        yield out, action_var
        cursor += batch

