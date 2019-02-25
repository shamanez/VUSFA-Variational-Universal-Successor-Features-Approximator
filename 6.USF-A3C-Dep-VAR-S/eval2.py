#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import sys
import os

from network import ActorCriticFFNetwork
from training_thread import A3CTrainingThread
from scene_loader import THORDiscreteEnvironment as Environment

from utils.ops import sample_action

from constants import ACTION_SIZE
from constants import CHECKPOINT_DIR
from constants import NUM_EVAL_EPISODES
from constants import VERBOSE

from constants import TASK_TYPE
# from constants import TASK_LIST
#180,408,676,468
skipStep=1
NUM_EVAL_EPISODES = 10
MAX_STEPS = 500



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pandas as pd

# TASK_LIST = {
#   'bathroom_02'    : map(str, range(0, 180,skipStep)),
#   'bedroom_04'     : map(str, range(0, 408,skipStep)),
#   'kitchen_02'     : map(str, range(0, 676,skipStep)),#676
#   'living_room_08' : map(str, range(0, 468,skipStep))
# }

TASK_LIST = {
  'bathroom_02'    : map(str, range(0, 180,skipStep)),
 'bedroom_04'     : map(str, range(0, 408,skipStep)),
  'kitchen_02'     : map(str, range(0, 676,skipStep)),#676
  'living_room_08' : map(str, range(0, 468,skipStep))
}



if __name__ == '__main__':

  device = "/cpu:0" # use CPU for display tool
  network_scope = TASK_TYPE
  list_of_tasks = TASK_LIST
  scene_scopes = list_of_tasks.keys()

  global_network = ActorCriticFFNetwork(action_size=ACTION_SIZE,
                                        device=device,
                                        network_scope=network_scope,
                                        scene_scopes=scene_scopes)

  sess = tf.Session()
  init = tf.global_variables_initializer()
  sess.run(init)

  saver = tf.train.Saver(global_network.get_vars())
  checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)

  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("checkpoint loaded: {}".format(checkpoint.model_checkpoint_path))
  else:
    print("Could not find old checkpoint")

  resultList = []
  allResultList = []
  scene_stats = dict()
  for scene_scope in scene_scopes:

    scene_stats[scene_scope] = []
    for task_scope in list_of_tasks[scene_scope]:

      print(task_scope)
      print("______________________________________________")

      env = Environment({
        'scene_name': scene_scope,
        'terminal_state_id': int(task_scope)
      })
      ep_rewards = []
      ep_lengths = []
      ep_collisions = []

      scopes = [network_scope, scene_scope, task_scope]

      for i_episode in range(NUM_EVAL_EPISODES):

        env.reset()
        terminal = False
        ep_reward = 0
        ep_collision = 0
        ep_t = 0

        while not terminal:

          usf_s_g = global_network.run_usf(sess, env.s_t, env.target, scopes)

          pi_values = global_network.run_policy(sess, env.s_t, env.target,usf_s_g, scopes)
          action = sample_action(pi_values)
          env.step(action)
          env.update()

          terminal = env.terminal
          if ep_t == MAX_STEPS: break
          if env.collided: ep_collision += 1
          ep_reward += env.reward
          ep_t += 1

        ep_lengths.append(ep_t)
        ep_rewards.append(ep_reward)
        ep_collisions.append(ep_collision)
        rs = [scene_scope,task_scope,ep_reward,ep_t,ep_collision]
        allResultList.append(rs)
        if VERBOSE: print("episode #{} ends after {} steps".format(i_episode, ep_t))

      print('evaluation: %s %s' % (scene_scope, task_scope))
      print('mean episode reward: %.2f' % np.mean(ep_rewards))
      print('mean episode length: %.2f' % np.mean(ep_lengths))
      print('mean episode collision: %.2f' % np.mean(ep_collisions))
      results = [scene_scope,task_scope,np.mean(ep_rewards),np.mean(ep_lengths),np.mean(ep_collisions)]
      resultList.append(results)

      scene_stats[scene_scope].extend(ep_lengths)
      pd.DataFrame(resultList).to_csv('intermediate-eval2.csv')
      #break
    #break

print('\nResults (average trajectory length):')
df = pd.DataFrame(resultList)
df.columns = ["scene_scope","task_scope","ep_rewards","ep_lengths","ep_collisions"]
df.to_csv('eval2.csv')
pd.DataFrame(allResultList).to_csv('eval2-allResults.csv')
for scene_scope in scene_stats:
  print('%s: %.2f steps'%(scene_scope, np.mean(scene_stats[scene_scope])))

