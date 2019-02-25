# -*- coding: utf-8 -*-
import sys
import h5py
import json
import numpy as np
import random
import skimage.io
from skimage.transform import resize
from constants import ACTION_SIZE
from constants import SCREEN_WIDTH
from constants import SCREEN_HEIGHT
from constants import HISTORY_LENGTH

import cv2

from PIL import Image
import pdb

class THORDiscreteEnvironment(object):

  def __init__(self, config=dict()):



    # configurations
    self.scene_name          = config.get('scene_name', 'bedroom_04')
    self.random_start        = config.get('random_start', True)
    self.n_feat_per_locaiton = config.get('n_feat_per_locaiton', 1) # 1 for no sampling
    self.terminal_state_id   = config.get('terminal_state_id', 0)

    self.h5_file_path = config.get('h5_file_path', 'data/%s.h5'%self.scene_name)
    self.h5_file      = h5py.File(self.h5_file_path, 'r')

    self.locations   = self.h5_file['location'][()]
    self.rotations   = self.h5_file['rotation'][()]
    self.n_locations = self.locations.shape[0]

    self.terminals = np.zeros(self.n_locations)
    self.terminals[self.terminal_state_id] = 1
    self.terminal_states, = np.where(self.terminals)

    self.transition_graph = self.h5_file['graph'][()]
    self.shortest_path_distances = self.h5_file['shortest_path_distance'][()]

    self.history_length = HISTORY_LENGTH
    self.screen_height  = SCREEN_HEIGHT
    self.screen_width   = SCREEN_WIDTH

    # we use pre-computed fc7 features from ResNet-50
    # self.s_t = np.zeros([self.screen_height, self.screen_width, self.history_length])
    self.s_t      = np.zeros([84,84,3])#, self.history_length])
    self.s_t1     = np.zeros_like(self.s_t)
    self.s_target = self._tiled_state(self.terminal_state_id)



    self.reset()


  # public methods

  def reset(self):
   

    # randomize initial state
    while True:
      k = random.randrange(self.n_locations)
      min_d = np.inf
      # check if target is reachable
      for t_state in self.terminal_states:
        dist = self.shortest_path_distances[k][t_state]
        min_d = min(min_d, dist)
      # min_d = 0  if k is a terminal state
      # min_d = -1 if no terminal state is reachable from k
      if min_d > 0: break

    # reset parameters
    self.current_state_id = k
    self.s_t = self._tiled_state(self.current_state_id)


    self.reward   = 0
    self.collided = False
    self.terminal = False

  def step(self, action):
    assert not self.terminal, 'step() called in terminal state'
    k = self.current_state_id

    if self.transition_graph[k][action] != -1:
      self.current_state_id = self.transition_graph[k][action]
      if self.terminals[self.current_state_id]:
        self.terminal = True
        self.collided = False
      else:
        self.terminal = False
        self.collided = False
    else:
      self.terminal = False
      self.collided = True
      

    self.reward = self._reward(self.terminal, self.collided)
    self.s_t1 = self.observation

  def update(self):
    self.s_t = self.s_t1

  # private methods

  def _tiled_state(self, state_id):
    k = random.randrange(self.n_feat_per_locaiton)
    I= self.h5_file['observation'][state_id]
    img_o = I#cv2.cvtColor(I, cv2.COLOR_RGB2GRAY )
    
    img_r = cv2.resize(img_o, (84, 84))/255#[255.,255.,255.]
    #f= img_r[:,:,:,np.newaxis]

    return img_r#np.tile(f, (1, self.history_length))

  def _reward(self, terminal, collided):
    # positive reward upon task completion
    if terminal: return 10.0
    # time penalty or collision penalty
    return -0.1 if collided else -0.01

  # properties

  @property
  def action_size(self):
    # move forward/backward, turn left/right for navigation
    return ACTION_SIZE 

  @property
  def action_definitions(self):
    action_vocab = ["MoveForward", "RotateRight", "RotateLeft", "MoveBackward"]
    return action_vocab[:ACTION_SIZE]

  @property
  def observation(self):
    new_image=self.h5_file['observation'][self.current_state_id]
    img_g = new_image#v2.cvtColor(new_image, cv2.COLOR_RGB2GRAY )

    img_s = cv2.resize(img_g, (84, 84))/255 #scaled new image
    #img_s=img_s[:,:,:,np.newaxis]
    return img_s 

  @property
  def state(self):
    # read from hdf5 cache
    k = random.randrange(self.n_feat_per_locaiton)
    return self.h5_file['resnet_feature'][self.current_state_id][k][:,np.newaxis]

  @property
  def target(self):
    return self.s_target

  @property
  def x(self):
    return self.locations[self.current_state_id][0]

  @property
  def z(self):
    return self.locations[self.current_state_id][1]

  @property
  def r(self):
    return self.rotations[self.current_state_id]

if __name__ == "__main__":
  scene_name = 'bedroom_04'

  env = THORDiscreteEnvironment({
    'random_start': True,
    'scene_name': scene_name,
    'h5_file_path': 'data/%s.h5'%scene_name
  })