# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pdb

# Actor-Critic Network Base Class
# The policy network and value network architecture
# should be implemented in a child class of this one
class ActorCriticNetwork(object):
  def __init__(self,
               action_size,
               device="/cpu:0"):
    self._device = device
    self._action_size = action_size

  def prepare_loss(self, entropy_beta, scopes):

    # drop task id (last element) as all tasks in
    # the same scene share the same output branch
    scope_key = self._get_key(scopes[:-1])

 

    with tf.device(self._device):
      # taken action (input for policy)
      self.a = tf.placeholder("float", [None, self._action_size])
    
      # temporary difference (R-V) (input for policy)
      self.td = tf.placeholder("float", [None])

      # avoid NaN with clipping when value in pi becomes zero
      log_pi = tf.log(tf.clip_by_value(self.pi[scope_key], 1e-20, 1.0))

      # policy entropy
      entropy = -tf.reduce_sum(self.pi[scope_key] * log_pi, axis=1)

      # policy loss (output)
      policy_loss = - tf.reduce_sum(tf.reduce_sum(tf.multiply(log_pi, self.a), axis=1) * self.td + entropy * entropy_beta)

      # R (input for value)
      self.r = tf.placeholder("float", [None])

      #Discounted USF return
      self.return_usf = tf.placeholder("float", [None,512])

      # value loss (output)
      # learning rate for critic is half of actor's
      value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v[scope_key])

      SR_loss = 0.0005*tf.losses.mean_squared_error(self.return_usf ,self.usf[scope_key])

      self.summary_SR_loss=SR_loss

      self.summary_Value_loss=value_loss  

      # gradienet of policy and value are summed up
      self.total_loss = policy_loss + value_loss + SR_loss

  def run_policy_and_value(self, sess, s_t, task):
    raise NotImplementedError()

  def run_policy(self, sess, s_t, task):
    raise NotImplementedError()

  def run_value(self, sess, s_t, task):
    raise NotImplementedError()

  def get_vars(self):
    raise NotImplementedError()

  def sync_from(self, src_netowrk, name=None):
    src_vars = src_netowrk.get_vars()
    dst_vars = self.get_vars()

    local_src_var_names = [self._local_var_name(x) for x in src_vars]
    local_dst_var_names = [self._local_var_name(x) for x in dst_vars]

    # keep only variables from both src and dst
    src_vars = [x for x in src_vars
      if self._local_var_name(x) in local_dst_var_names]
    dst_vars = [x for x in dst_vars
      if self._local_var_name(x) in local_src_var_names]

    sync_ops = []

    with tf.device(self._device):
      with tf.name_scope(name, "ActorCriticNetwork", []) as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)

  # variable (global/scene/task1/W_fc:0) --> scene/task1/W_fc:0
  def _local_var_name(self, var):
    return '/'.join(var.name.split('/')[1:])

  # weight initialization based on muupan's code
  # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
  def _fc_weight_variable(self, shape, name='W_fc'):
    input_channels = shape[0]
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _fc_bias_variable(self, shape, input_channels, name='b_fc'):
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv_weight_variable(self, shape, name='W_conv'):
    w = shape[0]
    h = shape[1]

    input_channels = shape[2]
    output_channels=shape[3]
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv_bias_variable(self, shape, w, h, input_channels, name='b_conv'):
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

  def _get_key(self, scopes):
    return '/'.join(scopes)

# Actor-Critic Feed-Forward Network
class ActorCriticFFNetwork(ActorCriticNetwork):
  """
    Implementation of the target-driven deep siamese actor-critic network from [Zhu et al., ICRA 2017]
    We use tf.variable_scope() to define domains for parameter sharing
  """
  def __init__(self,
               action_size,
               device="/cpu:0",
               network_scope="network",
               scene_scopes=["scene"]):
    ActorCriticNetwork.__init__(self, action_size, device)

    self.pi = dict()
    self.v = dict()
    self.state_rep=dict()
    self.usf = dict()

    ###############State
    self.W_k1=dict()
    self.b_k1=dict()


    self.W_k2=dict()
    self.b_k2=dict()


    self.W_k3=dict()
    self.b_k3=dict()

    self.w_fc_cnn=dict()
    self.b_fc_cnn=dict()

    self.w_fc_cnn_f=dict()
    self.b_fc_cnn_f=dict()

    #####################Goal
    self.W_k1_g=dict()
    self.b_k1_g=dict()


    self.W_k2_g=dict()
    self.b_k2_g=dict()


    self.W_k3_g=dict()
    self.b_k3_g=dict()

    self.w_fc_cnn_g=dict()
    self.b_fc_cnn_g=dict()

    self.w_fc_cnn_g_f=dict()
    self.b_fc_cnn_g_f=dict()
    #######################

    ########################Rewards
    self.W_fc1_rw = dict()
    self.b_fc1_rw = dict()

    self.W_fc2_rw = dict()
    self.b_fc2_rw = dict()

    #######################

    ####################Outputs

    self.W_fc2=dict()
    self.b_fc2=dict()

    self.W_fc3=dict()
    self.b_fc3=dict()

    self.W_policy = dict()
    self.b_policy = dict()

    self.W_usf = dict()
    self.b_usf = dict()

    self.lstm = dict()
    self.lstm_state = dict()
    self.state_out=dict()

    self.W_lstm = dict()
    self.b_lstm = dict()
    #########################################

    

    with tf.device(self._device):

      # state (input)
      self.s = tf.placeholder("float",[None,84,84,3])

      # target (input)
      self.t = tf.placeholder("float",[None,84,84,3])

      #LSTM stuff

      self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, 512])
      self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, 512])
      self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state0,
                                                              self.initial_lstm_state1)

      # place holder for LSTM unrolling time step size.
      self.step_size = tf.placeholder(tf.float32, [1])

      with tf.variable_scope(network_scope):
        # network key
        key = network_scope

        # flatten input
        self.s_flat = self.s
        self.t_flat =self.t        

        #For State###############################################################################
        self.W_k1[key] = self._conv_weight_variable([8,8,3,32])  
        self.b_k1[key] = self._conv_bias_variable([32],8,8,3)

        self.W_k2[key] = self._conv_weight_variable([4, 4,32,64])  
        self.b_k2[key] = self._conv_bias_variable([64],4,4,32)


        #Shared Siamese Layer_cnn_3_var
        self.W_k3[key] = self._conv_weight_variable([3,3,64,128])  
        self.b_k3[key] = self._conv_bias_variable([128],3,3,64)

        #Fc Layer after last CNN
        self.w_fc_cnn[key] = self._fc_weight_variable([2048, 512])
        self.b_fc_cnn[key] = self._fc_bias_variable([512], 2048)

        #Fc Layer after last CNN_f
        self.w_fc_cnn_f[key] = self._fc_weight_variable([512, 512])
        self.b_fc_cnn_f[key] = self._fc_bias_variable([512], 512)
        ###########################################################################################


        #For Goaol####################################################################

        self.W_k1_g[key] = self._conv_weight_variable([8,8,3,32])  
        self.b_k1_g[key] = self._conv_bias_variable([32],8,8,3)

        self.W_k2_g[key] = self._conv_weight_variable([4, 4,32,64])  
        self.b_k2_g[key] = self._conv_bias_variable([64],4,4,32)


        #Shared Siamese Layer_cnn_3_var
        self.W_k3_g[key] = self._conv_weight_variable([3,3,64,128])  
        self.b_k3_g[key] = self._conv_bias_variable([128],3,3,64)

        #Fc Layer after last CNN
        self.w_fc_cnn_g[key] = self._fc_weight_variable([2048, 512])
        self.b_fc_cnn_g[key] = self._fc_bias_variable([512], 2048)

        #Fc Layer after last CNN
        self.w_fc_cnn_g_f[key] = self._fc_weight_variable([512, 512])
        self.b_fc_cnn_g_f[key] = self._fc_bias_variable([512], 512)
        #####################################################################

  

        #Convoluting the states
        ############################################################################################################
        CNN_s_layer_1=tf.nn.leaky_relu(self._conv2d(self.s_flat,self.W_k1[key],4)+self.b_k1[key])  # Out_Put = ?, 109, 109, 32
        CNN_s_layer_2=tf.nn.leaky_relu(self._conv2d(CNN_s_layer_1,self.W_k2[key],2)+self.b_k2[key]) #(?, 53, 53, 64)
        CNN_s_layer_3=tf.nn.leaky_relu(self._conv2d(CNN_s_layer_2,self.W_k3[key],2)+self.b_k3[key]) #( 13, 13, 64)

        CNN_s_flat=tf.contrib.layers.flatten(CNN_s_layer_3)
        s_flat_h = tf.nn.relu(tf.matmul(CNN_s_flat, self.w_fc_cnn[key]) + self.b_fc_cnn[key])
        s_flat = tf.matmul(s_flat_h,self.w_fc_cnn_f[key]) + self.b_fc_cnn_f[key]
        #############################################################################################################


        #convoluting the target
        ############################################################################################################
        CNN_t_layer_1=tf.nn.leaky_relu(self._conv2d(self.t_flat,self.W_k1_g[key],4)+self.b_k1_g[key])
        CNN_t_layer_2=tf.nn.leaky_relu(self._conv2d(CNN_t_layer_1,self.W_k2_g[key],2)+self.b_k2_g[key])
        CNN_t_layer_3=tf.nn.leaky_relu(self._conv2d(CNN_t_layer_2,self.W_k3_g[key],2)+self.b_k3_g[key])

        CNN_t_flat=tf.contrib.layers.flatten(CNN_t_layer_3)
        t_flat_h = tf.nn.relu(tf.matmul(CNN_t_flat , self.w_fc_cnn_g[key]) + self.b_fc_cnn_g[key])
        t_flat = tf.matmul(t_flat_h , self.w_fc_cnn_g_f[key]) + self.b_fc_cnn_g_f[key]
        ############################################################################################################

        #concat bothe from the target and the state
        h_fc1 = tf.concat(values=[s_flat, t_flat], axis=1)

       
        # shared fusion layer
        self.W_fc2[key] = self._fc_weight_variable([1024, 512])
        self.b_fc2[key] = self._fc_bias_variable([512], 1024)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2[key]) + self.b_fc2[key])


        ###################################################################################
        #For the reward prediction network 

        self.W_fc1_rw[key] = self._fc_weight_variable([512, 512])
        self.b_fc1_rw[key] = self._fc_bias_variable([512], 512)

        self.W_fc2_rw[key] = self._fc_weight_variable([512, 512])
        self.b_fc2_rw[key] = self._fc_bias_variable([512], 512)

        reward_vector_hidden = tf.nn.relu(tf.matmul(t_flat, self.W_fc1_rw[key]) + self.b_fc1_rw[key])
        reward_vector = tf.matmul(reward_vector_hidden, self.W_fc2_rw[key]) + self.b_fc2_rw[key]

        

        for scene_scope in scene_scopes:
          # scene-specific key
          key = self._get_key([network_scope, scene_scope])

          with tf.variable_scope(scene_scope) as scope:

            # scene-specific adaptation layer
            self.W_fc3[key] = self._fc_weight_variable([512, 512])
            self.b_fc3[key] = self._fc_bias_variable([512], 512)
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2, self.W_fc3[key]) + self.b_fc3[key])


            #[batch_size, max_time, depth] Unolling (This will only work in the training (Bootstrapping))
            h_fc3_reshaped = tf.reshape(h_fc3 , [1, -1, 512]) #creating the batch as 4 consecative states(Unrolling)


       
            # lstm
            self.lstm[key] = tf.contrib.rnn.BasicLSTMCell(512, state_is_tuple=True)
            #tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=512)##tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(512,name="basic_lstm_cell")

            #During the training time this will unroll for the 5 time steps
            lstm_outputs, self.lstm_state[key]  = tf.nn.dynamic_rnn(self.lstm[key],
                                                        h_fc3_reshaped,
                                                        initial_state = self.initial_lstm_state,
                                                        sequence_length = self.step_size,
                                                        time_major = False,
                                                        scope = scope)



            lstm_outputs = tf.reshape(lstm_outputs, [-1,512]) #Converting them to a batch_size,1 to calculate loss


            # weight for policy output layer
            self.W_policy[key] = self._fc_weight_variable([512, action_size])
            self.b_policy[key] = self._fc_bias_variable([action_size], 512)

            # policy (output)
            pi_ = tf.matmul(lstm_outputs, self.W_policy[key]) + self.b_policy[key]
            self.pi[key] = tf.nn.softmax(pi_)


            # weight for USF output layer
            self.W_usf[key] = self._fc_weight_variable([512, 512])
            self.b_usf[key] = self._fc_bias_variable([512], 512)

            #usf(Output)
            usf_ = tf.matmul(lstm_outputs, self.W_usf[key]) + self.b_usf[key]
            self.usf[key] = usf_#tf.reshape(usf_, [-1])


            # value (output)
            v_ = tf.reduce_sum(tf.multiply(self.usf[key], reward_vector), 1, keepdims=True)

            self.v[key] = tf.reshape(v_, [-1])

            self.state_rep[key]=s_flat

            scope.reuse_variables()
            self.W_lstm[key] = tf.get_variable("basic_lstm_cell/kernel")
            self.b_lstm[key] = tf.get_variable("basic_lstm_cell/bias")


            self.reset_state()


       


  def reset_state(self):
    self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, 512]),
                                                        np.zeros([1, 512]))


  def run_policy_and_value(self, sess, state, target,scopes):
    k = self._get_key(scopes[:2])


    # This run_policy_and_value() is used when forward propagating.
    # so the step size is 1.
    pi_out, v_out,usf, self.lstm_state_out = sess.run( [self.pi[k], self.v[k], self.usf[k],self.lstm_state[k]],
                                                   feed_dict = {self.s : [state], self.t: [target],
                                                                self.initial_lstm_state0 : self.lstm_state_out[0],
                                                                self.initial_lstm_state1 : self.lstm_state_out[1],
                                                                self.step_size : [1]} )

    return pi_out[0],v_out[0],usf[0]
      


  def run_policy(self, sess, state, target,scopes):
    k = self._get_key(scopes[:2])
    # This run_policy() is used for displaying the result with display tool.    
    pi_out, self.lstm_state_out = sess.run( [self.pi[k], self.lstm_state[k]],
                                            feed_dict = {self.s : [state], self.t: [target],
                                                         self.initial_lstm_state0 : self.lstm_state_out[0],
                                                         self.initial_lstm_state1 : self.lstm_state_out[1],
                                                         self.step_size : [1]} )
                                            
    return pi_out[0]


  def run_value(self, sess, state,target,scopes):
    k = self._get_key(scopes[:2])
    # This run_value() is used for calculating V for bootstrapping at the 
    # end of LOCAL_T_MAX time step sequence.
    # When next sequcen starts, V will be calculated again with the same state using updated network weights,
    # so we don't update LSTM state here.
    prev_lstm_state_out = self.lstm_state_out
    v_out, _ = sess.run( [self.v[k], self.lstm_state],
                         feed_dict = {self.s : [state], self.t: [target],
                                      self.initial_lstm_state0 : self.lstm_state_out[0],
                                      self.initial_lstm_state1 : self.lstm_state_out[1],
                                      self.step_size : [1]} )
    
    # roll back lstm state
    self.lstm_state_out = prev_lstm_state_out
    return v_out[0]

  def run_usf(self, sess, state, target,scopes):
    k = self._get_key(scopes[:2])
    prev_lstm_state_out = self.lstm_state_out
    usf_out = sess.run( self.usf[k], feed_dict = {self.s : [state], self.t: [target],
      self.initial_lstm_state0 : self.lstm_state_out[0],
                                      self.initial_lstm_state1 : self.lstm_state_out[1],
                                      self.step_size : [1]} )

    # roll back lstm state
    self.lstm_state_out = prev_lstm_state_out

    return usf_out[0]

  def run_state(self, sess, state, scopes):
    k = self._get_key(scopes[:2])
    state_out = sess.run( self.state_rep[k], feed_dict = {self.s : [state]} )
    return state_out[0]

  def get_vars(self):
    var_list = [
      self.W_k1,self.b_k1,
      self.W_k2,self.b_k2,
      self.W_k3,self.b_k3,
      self.w_fc_cnn,self.b_fc_cnn,
      self.w_fc_cnn_f,self.b_fc_cnn_f,
      ###################################
      self.W_k1_g,self.b_k1_g,
      self.W_k2_g,self.b_k2_g,
      self.W_k3_g,self.b_k3_g,
      self.w_fc_cnn_g,self.b_fc_cnn_g,
      self.w_fc_cnn_g_f,self.b_fc_cnn_g_f,
      #################################
      self.W_fc2, self.b_fc2,
      self.W_fc3, self.b_fc3,
      self.W_policy, self.b_policy,

      self.W_fc1_rw,self.b_fc1_rw,
      self.W_fc2_rw,self.b_fc2_rw,
      self.W_usf,self.b_usf,
      self.W_lstm, self.b_lstm
    ]
    vs = []
    for v in var_list:
      vs.extend(v.values())
    return vs


  # def get_vars_pre_traied(self):
  #   var_list = [
  #     self.W_k1,self.b_k1,
  #     self.W_k2,self.b_k2,
  #     self.W_k3,self.b_k3,
  #     self.w_fc_cnn,self.b_fc_cnn,
  #     self.W_fc2, self.b_fc2,
  #     self.W_fc3, self.b_fc3
  #   ]
  #   vs = []
  #   for v in var_list:
  #     vs.extend(v.values())
  #   return vs
