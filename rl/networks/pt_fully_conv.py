import torch 
import torch.nn as nn 
from torch.autograd import Variable
import torch.functional as F

from pysc2.lib import actions
from pysc2.lib import features

import numpy as np

from rl.pre_processing import is_spatial_action, NUM_FUNCTIONS
from rl.pre_processing import screen_specs_sv, minimap_specs_sv, flat_specs_sv


def make_one_hot_1d(labels, dtype, C=2):
    '''
    Reference: https://lirnli.wordpress.com/2017/09/03/one-hot-encoding-in-pytorch/
    Parameters
    ----------
    labels : N, where N is batch size. 
    dtype: Cuda or not 
    C : number of classes in labels.
    
    Returns
    -------
    target : N x C
    '''
    out = Variable(dtype(labels.size(0), C).zero_())
    index = labels.contiguous().view(-1, 1).long()
    return out.scatter_(1, index, 1)


def make_one_hot_2d(labels, dtype, C=2):
    '''
    Reference: http://jacobkimmel.github.io/pytorch_onehot/
    Parameters
    ----------
    labels : torch.autograd.Variable of torch.LongTensor
        N x 1 x H x W, where N is batch size. 
    dtype: Cuda or not 
    C : number of classes in labels.
    
    Returns
    -------
    target : N x C x H x W
    '''
    one_hot = Variable(dtype(labels.size(0), C, labels.size(2), labels.size(3)).zero_())
    target = one_hot.scatter_(1, labels.long(), 1)
    return target


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class FullyConv(object):
    def __init__(self, args, supervised=False):
        self.args = args
        if not supervised:
            self.screen_specs = features.SCREEN_FEATURES
            self.minimap_specs = features.MINIMAP_FEATURES
        else:
            self.screen_specs = screen_specs_sv
            self.minimap_specs = minimap_specs_sv
        self.flat_specs = flat_specs_sv
        self.dtype = torch.FloatTensor
        self.atype = torch.LongTensor
        if args.cuda:
            self.dtype = torch.cuda.FloatTensor
            self.atype = torch.cuda.LongTensor

        self.embed_screen = self._init_embed_obs(self.screen_specs, self._embed_spatial)
        self.embed_minimap = self._init_embed_obs(self.minimap_specs, self._embed_spatial)
        self.embed_flat = self._init_embed_obs(self.flat_specs, self._embed_flat)

        self.screen_out = nn.DataParallel(nn.Sequential(
                self._conv2d_init(20, 8, stride=1, kernel_size=5, padding=2),
                nn.ReLU(True),
                self._conv2d_init(8, 16, stride=1, kernel_size=3, padding=1),
                nn.ReLU(True)))
        self.minimap_out = nn.DataParallel(nn.Sequential(
                self._conv2d_init(6, 12, stride=1, kernel_size=5, padding=2),
                nn.ReLU(True),
                self._conv2d_init(12, 16, stride=1, kernel_size=3, padding=1),
                nn.ReLU(True)))
        self.fc = nn.DataParallel(nn.Sequential(
                self._linear_init(43*64*64, 256),
                nn.ReLU(True)))
        self.value = nn.DataParallel(nn.Linear(in_features=256, out_features=1))
        self.fn_out = self._non_spatial_outputs(256, NUM_FUNCTIONS)
        self.non_spatial_outputs = self._init_non_spatial()
        self.spatial_outputs = self._init_spatial()


    def cuda(self):
        for k, v in self.embed_screen.items():
            v.cuda()
        for k, v in self.embed_minimap.items():
            v.cuda()
        for k, v in self.embed_flat.items():
            v.cuda()
        self.screen_out.cuda()
        self.minimap_out.cuda()
        self.fc.cuda()
        self.value.cuda()
        self.fn_out.cuda()
        for k, v in self.non_spatial_outputs.items():
            v.cuda()
        for k, v in self.spatial_outputs.items():
            v.cuda()


    def get_trainable_params(self, with_id=False):
        params = []
        ids = {}
        for k, v in self.embed_screen.items():
            ids['embed_screen:' + str(k)] = v
            params.extend(list(v.parameters()))
        for k, v in self.embed_minimap.items():
            ids['embed_minimap:' + str(k)] = v
            params.extend(list(v.parameters()))
        for k, v in self.embed_flat.items():
            ids['embed_flat:' + str(k)] = v
            params.extend(list(v.parameters()))
        ids['screen_out:0'] = self.screen_out
        params.extend(list(self.screen_out.parameters()))
        ids['minimap_out:0'] = self.minimap_out
        params.extend(list(self.minimap_out.parameters()))
        ids['fc:0'] = self.fc
        params.extend(list(self.fc.parameters()))
        ids['value:0'] = self.value
        params.extend(list(self.value.parameters()))
        ids['fn_out:0'] = self.fn_out
        params.extend(list(self.fn_out.parameters()))
        for k, v in self.non_spatial_outputs.items():
            ids['non_spatial_outputs:' + str(k)] = v
            params.extend(list(v.parameters()))
        for k, v in self.spatial_outputs.items():
            ids['spatial_outputs:' + str(k)] = v
            params.extend(list(v.parameters()))
        if not with_id: 
            return params
        else:
            return ids


    def log_grads(self, logger, i):
        params = self.get_trainable_params()
        for index, p in enumerate(params):   
            # logger.scalar_summary(str(index), p.grad.data.mean(), i)
            print(str(index), p.grad.data.mean())


    def _init_non_spatial(self):
        out = {}
        for arg_type in actions.TYPES:
            if not is_spatial_action[arg_type]:
                out[arg_type.id] = self._non_spatial_outputs(256, arg_type.sizes[0])
        return out


    def _init_spatial(self):
        out = {}
        for arg_type in actions.TYPES:
            if is_spatial_action[arg_type]:
                out[arg_type.id] = self._spatial_outputs(43)
        return out


    def _spatial_outputs(self, in_):
        return nn.DataParallel(nn.Sequential(
                    nn.Conv2d(in_channels=in_, out_channels=1, stride=1, kernel_size=1),
                    Flatten(),
                    nn.Softmax(dim=1)))


    def _non_spatial_outputs(self, in_, out_):
        return nn.DataParallel(nn.Sequential(nn.Linear(in_, out_),
                               nn.Softmax(dim=1)))


    def _conv2d_init(self, in_, out_, stride, kernel_size, padding):
        relu_gain = nn.init.calculate_gain('relu')
        conv = nn.Conv2d(in_, out_, stride=stride, 
                                kernel_size=kernel_size, padding=padding)
        conv.weight.data.mul_(relu_gain)
        return nn.DataParallel(conv)


    def _linear_init(self, in_, out_):
        relu_gain = nn.init.calculate_gain('relu')
        linear = nn.Linear(in_, out_)
        linear.weight.data.mul_(relu_gain)
        return nn.DataParallel(linear)


    def _init_embed_obs(self, spec, embed_fn):
        """
            Define network architectures
            Each input channel is processed by a Sequential network
        """
        out_sequence = {}
        for s in spec:
            if s.type == features.FeatureType.CATEGORICAL:
                dims = dims = np.round(np.log2(s.scale)).astype(np.int32).item()
                dims = max(dims, 1)
                sequence = nn.DataParallel(nn.Sequential(
                            embed_fn(s.scale, dims), 
                            nn.ReLU(True)))
                out_sequence[s.index] = sequence
        return out_sequence


    def _embed_spatial(self, in_, out_):
        return self._conv2d_init(in_, out_, kernel_size=1, stride=1, padding=0)


    def _embed_flat(self, in_, out_):
        return self._linear_init(in_, out_)


    def _log_transform(self, x, scale):
        return torch.log(8 * x / scale + 1)


    def _embed_obs(self, obs, spec, networks, one_hot):
        """
            Embed observation channels
        """
        # Channel dimension is 1
        feats = torch.chunk(obs, len(spec), dim=1)
        out_list = []
        for feat, s in zip(feats, spec):
            if s.type == features.FeatureType.CATEGORICAL:
                dims = np.round(np.log2(s.scale)).astype(np.int32).item()
                dims = max(dims, 1)
                indices = one_hot(feat, self.dtype, C=s.scale)
                out = networks[s.index](indices.float())
            elif s.type == features.FeatureType.SCALAR:
                out = self._log_transform(feat, s.scale)
            else:
                raise NotImplementedError
            out_list.append(out)
        # Channel dimension is 1
        return torch.cat(out_list, 1)    
        
        '''
        for s in spec:
            feat = feats[s.index]
            if s.type == features.FeatureType.CATEGORICAL:
                dims = np.round(np.log2(s.scale)).astype(np.int32).item()
                dims = max(dims, 1)
                indices = one_hot(feat, self.dtype, C=s.scale)
                out = networks[s.index](indices.float())
            elif s.type == features.FeatureType.SCALAR:
                out = self._log_transform(feat, s.scale)
            else:
                raise NotImplementedError
            out_list.append(out)
        # Channel dimension is 1
        return torch.cat(out_list, 1)
        '''

    
    def forward(self, screen_input, minimap_input, flat_input):
        _, _, resolution, _ = screen_input.size()

        screen_emb = self._embed_obs(screen_input, self.screen_specs, self.embed_screen, make_one_hot_2d)
        minimap_emb = self._embed_obs(minimap_input, self.minimap_specs, self.embed_minimap, make_one_hot_2d)
        flat_emb = self._embed_obs(flat_input, self.flat_specs, self.embed_flat, make_one_hot_1d)

        screen_out = self.screen_out(screen_emb)
        minimap_out = self.minimap_out(minimap_emb)
        broadcast_out = flat_emb.unsqueeze(2).unsqueeze(3). \
                        expand(flat_emb.size(0), flat_emb.size(1), resolution, resolution)

        state_out = torch.cat([screen_out.float(), minimap_out.float(), broadcast_out.float()], dim=1)
        flat_out = state_out.view(state_out.size(0), -1)
        fc = self.fc(flat_out)

        value = self.value(fc).view(-1)
        fn_out = self.fn_out(fc)

        args_out = dict()
        for arg_type in actions.TYPES:
            if is_spatial_action[arg_type]:
                arg_out = self.spatial_outputs[arg_type.id](state_out)
            else:
                arg_out = self.non_spatial_outputs[arg_type.id](fc)
            args_out[arg_type] = arg_out
        policy = (fn_out, args_out)
        return policy, value


