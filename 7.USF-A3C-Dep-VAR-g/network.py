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

    scope_key_b = self._get_key(scopes[:-2])


 



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

      alpha=1e-8 

      #mus=tf.concat(values=[self.mus_s_loss[scope_key_b],self.mus_s_loss_t[scope_key_b]],axis=0)
      #sigmas=tf.concat(values=[self.sigmas_s_loss[scope_key_b],self.sigmas_s_loss_t[scope_key_b]],axis=0)

      mus_t=tf.expand_dims(self.mus_s_loss_t[scope_key_b][0],0)
      sigmas_t=tf.expand_dims(self.sigmas_s_loss_t[scope_key_b][0],0)

      mus_s=self.mus_s_loss[scope_key_b]
      sigmas_s=self.sigmas_s_loss[scope_key_b]


      mus=tf.concat(values=[mus_t,mus_s],axis=1)
      sigmas=tf.concat(values=[sigmas_t,sigmas_s],axis=1)






      kl_Divergence= 0.5* ((tf.square(mus) + tf.square(sigmas) - tf.log(tf.square(sigmas) + alpha) - 1 ))
      I_c= 0.2
      self.bottle_neck=tf.reduce_mean(kl_Divergence-I_c)
      # gradienet of policy and value are summed up
      self.total_loss = policy_loss + value_loss + SR_loss + self.beta[scope_key_b]*self.bottle_neck

  def run_policy_and_value(self, sess, s_t, task):
    raise NotImplementedError()

  def run_policy(self, sess, s_t, task):
    raise NotImplementedError()

  def run_value(self, sess, s_t, task):
    raise NotImplementedError()

  def run_usf(self, sess, s_t, task):
    raise NotImplementedError()

  def run_state(self, sess, s_t):
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

##################################################################################

  def sync_beta_from_glob(self, src_netowrk, name=None):
    src_vars = src_netowrk.get_beta_var()
    dst_vars = self.get_beta_var()

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


  def sync_beta_to_glob(self, src_netowrk, name=None):
    src_vars = self.get_beta_var()
    dst_vars = src_netowrk.get_beta_var()

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
  ##################################################################################

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

    self.mus_s_loss=dict()
    self.sigmas_s_loss=dict()

    self.mus_s_loss_t=dict()
    self.sigmas_s_loss_t=dict()

    self.assign_op=dict()

    self.beta=dict()

    self.W_fc1 = dict()
    self.b_fc1 = dict()

    self.W_fc1_siamese = dict()
    self.b_fc1_siamese = dict()



    self.W_fc2 = dict()
    self.b_fc2 = dict()

    self.W_fc3 = dict()
    self.b_fc3 = dict()

    self.W_fc1_rw = dict()
    self.b_fc1_rw = dict()

    self.W_fc1_s1 = dict()
    self.b_fc1_s1 = dict()

    self.W_fc1_r1 = dict()
    self.b_fc1_r1 = dict()

    self.W_policy = dict()
    self.b_policy = dict()

    self.W_policy_h = dict()
    self.b_policy_h = dict()

    self.W_value = dict()
    self.b_value = dict()


    self.W_usf = dict()
    self.b_usf = dict()

    with tf.device(self._device):

      # state (input)
      self.s = tf.placeholder("float", [None, 2048, 1])

      # target (input)
      self.t = tf.placeholder("float", [None, 2048, 1])


      # USF_appxi (input)
      self.usf_policy = tf.placeholder("float", [None, 512])

      #Bottleneck loss 

      self.KL_bottleneck_loss= tf.placeholder("float")



      with tf.variable_scope(network_scope):
        # network key
        key = network_scope


        self.beta[key]=tf.Variable(tf.zeros([]), name="beta") #self.bottleneck_loss

        beta_new=tf.maximum(0.0,self.beta[key]+1e-6 *self.KL_bottleneck_loss)



        self.assign_op[key]=tf.assign(self.beta[key],beta_new)


        

        # flatten input
        self.s_flat = tf.reshape(self.s, [-1, 2048])
        self.t_flat = tf.reshape(self.t, [-1, 2048])

        # shared siamese layer
        self.W_fc1[key] = self._fc_weight_variable([2048, 512])
        self.b_fc1[key] = self._fc_bias_variable([512], 2048)



        # shared siamese layer
        self.W_fc1_siamese[key] = self._fc_weight_variable([512, 512])
        self.b_fc1_siamese[key] = self._fc_bias_variable([512], 512)


        h_s_flat_1 = tf.nn.relu(tf.matmul(self.s_flat, self.W_fc1[key]) + self.b_fc1[key])

        h_s_flat_1_siamese = tf.nn.relu(tf.matmul(h_s_flat_1, self.W_fc1_siamese[key]) + self.b_fc1_siamese[key])

        # add one more fully connected layer with linear activation to be state_rep
        self.W_fc1_s1[key] = self._fc_weight_variable([512, 1024])
        self.b_fc1_s1[key] = self._fc_bias_variable([1024], 512)
        h_s_flat_KL = tf.matmul(h_s_flat_1_siamese, self.W_fc1_s1[key]) + self.b_fc1_s1[key]

        params_s=h_s_flat_KL.shape[-1]//2

        mus_s=h_s_flat_KL[:,:params_s]

        sigmas_s=h_s_flat_KL[:,params_s:]

    

        t=tf.keras.backend.shape(mus_s)[0]


        eps_s=tf.keras.backend.random_normal(shape=(t,512),mean=0,stddev=1)


        h_s_flat=mus_s+ sigmas_s*eps_s
        h_s_flat_real=mus_s	

        h_t_flat_1 = tf.nn.relu(tf.matmul(self.t_flat, self.W_fc1[key]) + self.b_fc1[key])

        # add one more fully connected layer with linear activation to be target rep
        h_t_flat_KL = tf.matmul(h_t_flat_1, self.W_fc1_s1[key]) + self.b_fc1_s1[key]

        mus_t=h_t_flat_KL[:,:params_s]

        sigmas_t=h_t_flat_KL[:,params_s:]

        eps_t=tf.keras.backend.random_normal(shape=(t,512),mean=0,stddev=1)

        h_t_flat=mus_t+ sigmas_t*eps_t
        h_t_flat_real=mus_t

        self.mus_s_loss[key]=mus_s
        self.sigmas_s_loss[key]=sigmas_s


        self.mus_s_loss_t[key]=mus_t
        self.sigmas_s_loss_t[key]=sigmas_t



        h_fc1 = tf.concat(values=[h_s_flat_real, h_t_flat_real], axis=1)
        #h_fc1 = tf.concat(values=[h_s_flat_real, h_t_flat], axis=1)

        # shared fusion layer
        self.W_fc2[key] = self._fc_weight_variable([1024, 512])
        self.b_fc2[key] = self._fc_bias_variable([512], 1024)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2[key]) + self.b_fc2[key])

        #one more fc before reward Prediction
        self.W_fc1_r1[key] = self._fc_weight_variable([512, 512])
        self.b_fc1_r1[key] = self._fc_bias_variable([512], 512)
        h_t_flat_2 = tf.nn.relu(tf.matmul(h_t_flat, self.W_fc1_r1[key]) + self.b_fc1_r1[key])

        # Reward Vector Prediction Layer
        self.W_fc1_rw[key] = self._fc_weight_variable([512, 512])
        self.b_fc1_rw[key] = self._fc_bias_variable([512], 512)
        reward_vector = tf.matmul(h_t_flat_2, self.W_fc1_rw[key]) + self.b_fc1_rw[key]
        #reward_vector = tf.matmul(h_t_flat, self.W_fc1_rw[key]) + self.b_fc1_rw[key]



        for scene_scope in scene_scopes:
          # scene-specific key
          key = self._get_key([network_scope, scene_scope])

          with tf.variable_scope(scene_scope):

            # scene-specific adaptation layer
            self.W_fc3[key] = self._fc_weight_variable([512, 512])
            self.b_fc3[key] = self._fc_bias_variable([512], 512)
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2, self.W_fc3[key]) + self.b_fc3[key])



            usf_policy_h=tf.concat(values=[h_fc3, self.usf_policy], axis=1)

            self.W_policy_h[key] = self._fc_weight_variable([1024, 512])
            self.b_policy_h[key] = self._fc_bias_variable([512], 1024)


            pi_h = tf.nn.relu(tf.matmul(usf_policy_h, self.W_policy_h[key]) + self.b_policy_h[key])


            # weight for policy output layer
            self.W_policy[key] = self._fc_weight_variable([512, 4])
            self.b_policy[key] = self._fc_bias_variable([4], 512)


            # policy (output)
            pi_=tf.nn.relu(tf.matmul(pi_h, self.W_policy[key]) + self.b_policy[key])
            self.pi[key] = tf.nn.softmax(pi_)


            # weight for USF output layer
            self.W_usf[key] = self._fc_weight_variable([512, 512])
            self.b_usf[key] = self._fc_bias_variable([512], 512)

            #usf(Output)
            usf_ = tf.matmul(h_fc3, self.W_usf[key]) + self.b_usf[key]
            self.usf[key] = usf_#tf.reshape(usf_, [-1])


            # value (output)
            v_ = tf.reduce_sum(tf.multiply(self.usf[key], reward_vector), 1, keepdims=True)

            self.v[key] = tf.reshape(v_, [-1])
            self.state_rep[key]=h_s_flat_real




  


  def run_policy_and_value(self, sess, state, target, usf,scopes):
    k = self._get_key(scopes[:2])
    pi_out, v_out = sess.run( [self.pi[k], self.v[k]], feed_dict = {self.s : [state], self.t: [target],self.usf_policy:[usf]} )
    return (pi_out[0], v_out[0])


  def run_beta_ass_op(self, sess,loss,scopes):
    k = self._get_key(scopes[:1])



    sess.run( [self.assign_op[k]], feed_dict = {self.KL_bottleneck_loss : loss} )

  def run_policy(self, sess, state, target, usf,scopes):
    k = self._get_key(scopes[:2])
    pi_out = sess.run( self.pi[k], feed_dict = {self.s : [state], self.t: [target],self.usf_policy:[usf]} )
    return pi_out[0]

  def run_value(self, sess, state, target, scopes):
    k = self._get_key(scopes[:2])
    v_out = sess.run( self.v[k], feed_dict = {self.s : [state], self.t: [target]} )
    return v_out[0]

  def run_usf(self, sess, state, target, scopes):
    k = self._get_key(scopes[:2])
    usf_out = sess.run( self.usf[k], feed_dict = {self.s : [state], self.t: [target]} )
    return usf_out[0]

  def run_state(self, sess, state, scopes):
    k = self._get_key(scopes[:2])
    state_out = sess.run( self.state_rep[k], feed_dict = {self.s : [state]} )
    return state_out[0]

  def get_vars(self):
    var_list = [
      self.W_fc1, self.b_fc1,
      self.W_fc1_siamese, self.b_fc1_siamese,
      self.W_fc2, self.b_fc2,
      self.W_fc3, self.b_fc3,
      self.W_policy_h, self.b_policy_h,
      self.W_policy, self.b_policy,
      #self.W_value, self.b_value
      self.W_fc1_s1, self.b_fc1_s1,
      self.W_fc1_r1, self.b_fc1_r1,
      self.W_fc1_rw,self.b_fc1_rw,
      self.W_usf,self.b_usf

    ]
    vs = []
    for v in var_list:
      vs.extend(v.values())
    return vs


  def get_beta_var(self):
    var_list = [

    self.beta
      
    ]
    vs = []
    for v in var_list:
      vs.extend(v.values())
    return vs


  def get_vars_pre(self):
    var_list = [
      #self.W_fc1, self.b_fc1,
      #self.W_fc1_siamese, self.b_fc1_siamese,
      self.W_fc2, self.b_fc2,
      self.W_fc3, self.b_fc3,
      self.W_policy_h, self.b_policy_h,
      self.W_policy, self.b_policy,
      #self.W_value, self.b_value
      self.W_fc1_s1, self.b_fc1_s1,
      self.W_fc1_r1, self.b_fc1_r1,
      self.W_fc1_rw,self.b_fc1_rw,
      self.W_usf,self.b_usf

    ]
    vs = []
    for v in var_list:
      vs.extend(v.values())
    return vs
