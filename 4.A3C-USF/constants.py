# -*- coding: utf-8 -*-

LOCAL_T_MAX = 5 # repeat step size
RMSP_ALPHA = 0.99 # decay parameter for RMSProp
RMSP_EPSILON = 0.1 # epsilon parameter for RMSProp
CHECKPOINT_DIR = 'checkpoints'
LOG_FILE = 'logs'
INITIAL_ALPHA_LOW = 1e-4    # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-2   # log_uniform high limit for learning rate

PARALLEL_SIZE = 32 # parallel thread size
ACTION_SIZE = 4 # action size

INITIAL_ALPHA_LOG_RATE = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
GAMMA = 0.99 # discount factor for rewards
ENTROPY_BETA = 0.01 # entropy regurarlization constant
MAX_TIME_STEP = 10.0 * 10**6 # 10 million frames
GRAD_NORM_CLIP = 40.0 # gradient norm clipping
USE_GPU = True # To use GPU, set True
VERBOSE = True

SCREEN_WIDTH = 84
SCREEN_HEIGHT = 84
HISTORY_LENGTH = 4

NUM_EVAL_EPISODES = 100 # number of episodes for evaluation

TASK_TYPE = 'navigation' # no need to change
# keys are scene names, and values are a list of location ids (navigation targets)
TASK_LIST = {
  'bathroom_02'    : ['26', '37', '43', '53', '69','5', '20', '15', '10', '8', '20', '15', '10', '8']}
#   ,
#   'bedroom_04'     : ['134', '264', '320', '384', '387'],
#   'kitchen_02'     : ['90', '136', '157', '207', '329'],
#   'living_room_08' : ['92', '135', '193', '228', '254']
# }
