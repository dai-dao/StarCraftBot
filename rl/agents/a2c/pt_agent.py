import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.autograd import Variable
import torch.functional as F
from torch.distributions import Categorical

import os
import numpy as np 

from rl.networks.pt_fully_conv import FullyConv
from rl.pre_processing import Preprocessor


class A2CAgent(object):
    def __init__(self, args):
        self.args = args
        self.dtype = torch.FloatTensor
        self.atype = torch.LongTensor
        self.network = FullyConv(args)
        if args.cuda:
            self.dtype = torch.cuda.FloatTensor
            self.atype = torch.cuda.LongTensor
            self.network.cuda()
        self.optimizer = optim.Adam(self.network.get_trainable_params(), lr=args.lr)


    def step(self, last_obs):
        last_obs_var = self._make_var(last_obs)
        policy, value = self.network.forward(last_obs_var["screen"], 
                                        last_obs_var["minimap"], last_obs_var["flat"])
        available_actions = last_obs_var["available_actions"]
        samples = self._sample_actions(available_actions, policy)
        return samples, value.cpu().data.numpy()


    def get_value(self, last_obs):
        _, value_estimate = self.step(last_obs)
        return value_estimate


    def train(self, obs, actions, returns, advs, summary=False):
        obs_var = self._make_var(obs)
        returns_var = Variable(self.dtype(returns))
        advs_var = Variable(self.dtype(advs))
        actions_var = self._make_var_actions(actions)
        available_actions = obs_var["available_actions"]
        policy, value = self.network.forward(obs_var["screen"], \
                                            obs_var["minimap"], obs_var["flat"])
        log_probs = self._compute_policy_log_probs(available_actions, policy, actions_var)
        policy_loss = -(advs_var * log_probs).mean()
        value_loss = (returns_var - value).pow(2).mean()
        entropy = self._compute_policy_entropy(available_actions, policy, actions_var)
        loss = policy_loss + value_loss * self.args.value_loss_weight \
                - entropy * self.args.entropy_weight
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.network.get_trainable_params(), self.args.max_gradient_norm)
        self.optimizer.step()
        return None


    def log(self, logger, i):
        # self.network.log_params(logger, i)
        self.network.log_grads(logger, i)


    def save_checkpoint(self, epoch_count):
        print('Saving checkpoint at', self.args.save_dir + '.pth.tar')
        out = {}
        params = self.network.get_trainable_params(with_id=True)
        for k, v in params.items():
            out[k] = v.state_dict()
        out['epoch']  = epoch_count
        out['optimizer'] = self.optimizer.state_dict()
        torch.save(out, self.args.save_dir + '.pth.tar')


    def load_checkpoint(self):
        print('Loading checkpoint at', self.args.save_dir + '.pth.tar')
        loaded_params = torch.load(self.args.save_dir + '.pth.tar')
        params = self.network.get_trainable_params(with_id=True)
        for k, v in params.items():
            v.load_state_dict(loaded_params[k])
        self.optimizer.load_state_dict(loaded_params['optimizer'])
        current_epoch_count = loaded_params['epoch']
        return current_epoch_count


    def _make_var(self, input_dict):
        new_dict = {}
        for k, v in input_dict.items():
            new_dict[k] = Variable(self.dtype(v))
        return new_dict
    

    def _sample(self, probs):
        dist = Categorical(probs=probs)
        return dist.sample()


    def _mask_unavailable_actions(self, available_actions, fn_pi):
        fn_pi = fn_pi * available_actions
        fn_pi = fn_pi / fn_pi.sum(1, keepdim=True)
        return fn_pi

    
    def _sample_actions(self, available_actions, policy):
        fn_pi, arg_pis = policy
        fn_pi = self._mask_unavailable_actions(available_actions, fn_pi)  
        
        # Sample actions
        # Avoid the case where the sampled action is NOT available
        while True:
            fn_samples = self._sample(fn_pi)
            if (available_actions.gather(1, fn_samples.unsqueeze(1)) == 1).all():
                fn_samples = fn_samples.data.cpu().numpy()
                break

        arg_samples = dict()
        for arg_type, arg_pi in arg_pis.items():
            arg_samples[arg_type] = self._sample(arg_pi).data.cpu().numpy()
        return fn_samples, arg_samples


    def _make_var_actions(self, actions):
        n_id, arg_ids = actions 
        args_var = {}
        fn_id_var = Variable(self.atype(n_id))
        for k, v in arg_ids.items():
            args_var[k] = Variable(self.atype(v))
        return fn_id_var, args_var


    def _compute_policy_log_probs(self, available_actions, policy, actions_var):
        def logclip(x):
            return torch.log(torch.clamp(x, 1e-12, 1.0))

        def compute_log_probs(probs, labels):
            new_labels = labels.clone()
            new_labels[new_labels < 0] = 0
            selected_probs = probs.gather(1, new_labels.unsqueeze(1))
            out = logclip(selected_probs)
            # Log of 0 will be 0
            # out[selected_probs == 0] = 0
            return out.view(-1)

        fn_id, arg_ids = actions_var
        fn_pi, arg_pis = policy
        fn_pi = self._mask_unavailable_actions(available_actions, fn_pi)
        fn_log_prob = compute_log_probs(fn_pi, fn_id)

        log_prob = fn_log_prob
        for arg_type in arg_ids.keys():
            arg_id = arg_ids[arg_type]
            arg_pi = arg_pis[arg_type]
            arg_log_prob = compute_log_probs(arg_pi, arg_id)

            arg_id_masked = arg_id.clone()
            arg_id_masked[arg_id_masked != -1] = 1
            arg_id_masked[arg_id_masked == -1] = 0
            arg_log_prob = arg_log_prob * arg_id_masked.float()
            log_prob = log_prob + arg_log_prob
        return log_prob


    def _compute_policy_entropy(self, available_actions, policy, actions_var):
        def logclip(x):
            return torch.log(torch.clamp(x, 1e-12, 1.0))
        
        def compute_entropy(probs):
            return -(logclip(probs) * probs).sum(-1)
        
        _, arg_ids = actions_var    
        fn_pi, arg_pis = policy
        fn_pi = self._mask_unavailable_actions(available_actions, fn_pi)

        entropy = compute_entropy(fn_pi).mean()
        for arg_type in arg_ids.keys():
            arg_id = arg_ids[arg_type]
            arg_pi = arg_pis[arg_type]

            batch_mask = arg_id.clone()
            batch_mask[batch_mask != -1] = 1
            batch_mask[batch_mask == -1] = 0
            # Reference: https://discuss.pytorch.org/t/how-to-use-condition-flow/644/4
            if (batch_mask == 0).all():
                arg_entropy = (compute_entropy(arg_pi) * 0.0).sum()
            else:
                arg_entropy = (compute_entropy(arg_pi) * batch_mask.float()).sum() / batch_mask.float().sum()
            entropy = entropy + arg_entropy
        return entropy