import numpy as np
from collections import namedtuple
from pysc2.lib import actions
from pysc2.lib import features


NUM_FUNCTIONS = len(actions.FUNCTIONS)
NUM_PLAYERS = features.SCREEN_FEATURES.player_id.scale


is_spatial_action = {}
for name, arg_type in actions.TYPES._asdict().items():
  # HACK: we should infer the point type automatically
  is_spatial_action[arg_type] = name in ['minimap', 'screen', 'screen2']


def stack_ndarray_dicts(lst, axis=0):
  """Concatenate ndarray values from list of dicts
  along new axis."""
  res = {}
  for k in lst[0].keys():
    res[k] = np.stack([d[k] for d in lst], axis=axis)
  return res


SVSpec = namedtuple('SVSpec', ['type', 'index', 'scale'])
screen_specs_sv = [SVSpec(features.FeatureType.SCALAR, 0 , 256.),
                   SVSpec(features.FeatureType.CATEGORICAL, 1, 4),
                   SVSpec(features.FeatureType.CATEGORICAL, 2, 2),
                   SVSpec(features.FeatureType.CATEGORICAL, 3, 2),
                   SVSpec(features.FeatureType.CATEGORICAL, 5, 5),
                   SVSpec(features.FeatureType.CATEGORICAL, 6, 1850),
                   SVSpec(features.FeatureType.SCALAR, 14, 16.),
                   SVSpec(features.FeatureType.SCALAR, 15, 256.)]


minimap_specs_sv = [SVSpec(features.FeatureType.SCALAR, 0 , 256.),
                    SVSpec(features.FeatureType.CATEGORICAL, 1 , 4),
                    SVSpec(features.FeatureType.CATEGORICAL, 2 , 2),
                    SVSpec(features.FeatureType.CATEGORICAL, 5 , 5)]


flat_specs_sv = [SVSpec(features.FeatureType.SCALAR, 0, 1.),
                 SVSpec(features.FeatureType.SCALAR, 1, 1.),
                 SVSpec(features.FeatureType.SCALAR, 2, 1.),
                 SVSpec(features.FeatureType.SCALAR, 3, 1.),
                 SVSpec(features.FeatureType.SCALAR, 4, 1.),
                 SVSpec(features.FeatureType.SCALAR, 5, 1.),
                 SVSpec(features.FeatureType.SCALAR, 6, 1.),
                 SVSpec(features.FeatureType.SCALAR, 7, 1.),
                 SVSpec(features.FeatureType.SCALAR, 8, 1.),
                 SVSpec(features.FeatureType.SCALAR, 9, 1.),
                 SVSpec(features.FeatureType.SCALAR, 10, 1.)]


class Preprocessor():
  """Compute network inputs from pysc2 observations.

  See https://github.com/deepmind/pysc2/blob/master/docs/environment.md
  for the semantics of the available observations.
  """   
  
  def __init__(self):
    self.screen_channels = len(features.SCREEN_FEATURES)
    self.minimap_channels = len(features.MINIMAP_FEATURES)
    self.flat_channels = len(flat_specs_sv)
    self.available_actions_channels = NUM_FUNCTIONS


  def get_input_channels(self):
    """Get static channel dimensions of network inputs."""
    return {
        'screen': self.screen_channels,
        'minimap': self.minimap_channels,
        'flat': self.flat_channels,
        'available_actions': self.available_actions_channels}


  def preprocess_obs(self, obs_list):
    return stack_ndarray_dicts(
        [self._preprocess_obs(o.observation) for o in obs_list])


  def _preprocess_obs_sv(self, obs):
        chosen_screen_index = [0, 1, 2, 3, 5, 6, 14, 15]
        chosen_minimap_index = [0, 1, 2, 5]
        screen = np.stack([obs['screen'][i] for i in chosen_screen_index], axis=0)
        minimap = np.stack([obs['minimap'][i] for i in chosen_minimap_index], axis=0)
        flat = np.concatenate([obs['player']])
        return {'screen': screen,
                'minimap': minimap,
                'flat': flat}


  def _preprocess_obs(self, obs):
    """Compute screen, minimap and flat network inputs from raw observations.
    """
    available_actions = np.zeros(NUM_FUNCTIONS, dtype=np.float32)
    available_actions[obs['available_actions']] = 1

    screen = self._preprocess_spatial(obs['screen'])
    minimap = self._preprocess_spatial(obs['minimap'])

    flat = np.concatenate([
        obs['player']])
        # TODO available_actions, control groups, cargo, multi select, build queue

    return {
        'screen': screen,
        'minimap': minimap,
        'flat': flat,
        'available_actions': available_actions}


  def _preprocess_spatial(self, spatial):
    # PYTORCH
    return spatial

    # TF
    # return np.transpose(spatial, [1, 2, 0])
