# -*- coding: utf-8 -*-

LOCAL_T_MAX = 5 # repeat step size
RMSP_ALPHA = 0.99 # decay parameter for RMSProp
RMSP_EPSILON = 0.1 # epsilon parameter for RMSProp
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_DIR_beta='checkpoints_b'
LOG_FILE = 'logs'
INITIAL_ALPHA_LOW = 1e-4    # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-2   # log_uniform high limit for learning rate

PARALLEL_SIZE =100#64 # parallel thread size
ACTION_SIZE = 4 # action size

INITIAL_ALPHA_LOG_RATE = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
GAMMA = 0.99 # discount factor for rewards
ENTROPY_BETA = 0.01 # entropy regurarlization constant
MAX_TIME_STEP = 100.0 * 10**6 # 10 million frames
GRAD_NORM_CLIP = 40.0 # gradient norm clipping
USE_GPU = True # To use GPU, set True
VERBOSE = True

SCREEN_WIDTH = 84
SCREEN_HEIGHT = 84
HISTORY_LENGTH = 1 

NUM_EVAL_EPISODES = 10 # number of episodes for evaluation

TASK_TYPE = 'navigation' # no need to change
# keys are scene names, and values are a list of location ids (navigation targets)

TASK_LIST = {
  'bathroom_02'    : ['26', '37', '43', '53', '69'],
  'bedroom_04'     : ['134', '264', '320', '384', '387'],
  'kitchen_02'     : ['368','90', '136','207', '157', '329'],
  'living_room_08' : ['193','92', '135', '206', '228', '254']
}

# TASK_LIST = {
#   'bathroom_02'    : ['150','26', '37', '43', '53', '69','72','96','65'],
#   'bedroom_04'     : ['23','225','91','407','289','293','134', '264', '320', '384', '387'],
#   'kitchen_02'     : ['90', '136', '157', '207', '329','116','260','240','256','89'],
#   'living_room_08' : ['92','366','266','185','217','191','447', '135']
# }


# TASK_LIST = {
#   'bathroom_02'    : ['150','26', '37', '43', '53', '69','40','72','96','65'],
#   'bedroom_04'     : ['23','209','225','91','79','407','289','293','134', '264', '320', '384', '387','407','52'],
#   'kitchen_02'     : ['90', '136', '157', '207', '329','116','260','240','256','89','373','292','332','373','377','596','643','408','643'],
#   'living_room_08' : ['92','366','266','185','217','191','447', '135', '193', '228', '254','133','100','8','145','191']
# }
