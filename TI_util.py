""" Wrapper functions for TensorFlow layers.

Author: Charles R. Qi
Date: November 2017
"""

import numpy as np
import tensorflow as tf

def _variable_on_cpu(name, shape, initializer, use_fp16=False):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if use_fp16 else tf.float32
  var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

  Returns:
    Variable Tensor
  """
  if use_xavier:
    initializer = tf.contrib.layers.xavier_initializer()
  else:
    initializer = tf.truncated_normal_initializer(stddev=stddev)
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 1D convolution with non-linear operation.

  Args:
    inputs: 3-D tensor variable BxLxC
    num_output_channels: int
    kernel_size: int
    scope: string
    stride: int
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    num_in_channels = inputs.get_shape()[-1].value
    kernel_shape = [kernel_size,
                    num_in_channels, num_output_channels]
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
    outputs = tf.nn.conv1d(inputs, kernel,
                           stride=stride,
                           padding=padding)
    biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)

    if bn:
      outputs = batch_norm_for_conv1d(outputs, is_training,
                                      bn_decay=bn_decay, scope='bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs




def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.get_shape()[-1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_in_channels, num_output_channels]
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      outputs = tf.nn.conv2d(inputs, kernel,
                             [1, stride_h, stride_w, 1],
                             padding=padding)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)

      if bn:
        outputs = batch_norm_for_conv2d(outputs, is_training,
                                        bn_decay=bn_decay, scope='bn')

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs


def conv2d_transpose(inputs,
                     num_output_channels,
                     kernel_size,
                     scope,
                     stride=[1, 1],
                     padding='SAME',
                     use_xavier=True,
                     stddev=1e-3,
                     weight_decay=0.0,
                     activation_fn=tf.nn.relu,
                     bn=False,
                     bn_decay=None,
                     is_training=None):
  """ 2D convolution transpose with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor

  Note: conv2d(conv2d_transpose(a, num_out, ksize, stride), a.shape[-1], ksize, stride) == a
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.get_shape()[-1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_output_channels, num_in_channels] # reversed to conv2d
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      
      # from slim.convolution2d_transpose
      def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
          dim_size *= stride_size

          if padding == 'VALID' and dim_size is not None:
            dim_size += max(kernel_size - stride_size, 0)
          return dim_size

      # caculate output shape
      batch_size = inputs.get_shape()[0].value
      height = inputs.get_shape()[1].value
      width = inputs.get_shape()[2].value
      out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
      out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
      output_shape = [batch_size, out_height, out_width, num_output_channels]

      outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape,
                             [1, stride_h, stride_w, 1],
                             padding=padding)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)

      if bn:
        outputs = batch_norm_for_conv2d(outputs, is_training,
                                        bn_decay=bn_decay, scope='bn')

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs

   

def conv3d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 3D convolution with non-linear operation.

  Args:
    inputs: 5-D tensor variable BxDxHxWxC
    num_output_channels: int
    kernel_size: a list of 3 ints
    scope: string
    stride: a list of 3 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    num_in_channels = inputs.get_shape()[-1].value
    kernel_shape = [kernel_d, kernel_h, kernel_w,
                    num_in_channels, num_output_channels]
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.conv3d(inputs, kernel,
                           [1, stride_d, stride_h, stride_w, 1],
                           padding=padding)
    biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)
    
    if bn:
      outputs = batch_norm_for_conv3d(outputs, is_training,
                                      bn_decay=bn_decay, scope='bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs

def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.0,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None):
  """ Fully connected layer with non-linear operation.
  
  Args:
    inputs: 2-D tensor BxN
    num_outputs: int
  
  Returns:
    Variable tensor of size B x num_outputs.
  """
  with tf.variable_scope(scope) as sc:
    num_input_units = inputs.get_shape()[-1].value
    weights = _variable_with_weight_decay('weights',
                                          shape=[num_input_units, num_outputs],
                                          use_xavier=use_xavier,
                                          stddev=stddev,
                                          wd=weight_decay)
    outputs = tf.matmul(inputs, weights)
    biases = _variable_on_cpu('biases', [num_outputs],
                             tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)
     
    if bn:
      outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D max pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.max_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs

def avg_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D avg pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.avg_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs


def max_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
  """ 3D max pooling.

  Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.max_pool3d(inputs,
                               ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1],
                               padding=padding,
                               name=sc.name)
    return outputs

def avg_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
  """ 3D avg pooling.

  Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.avg_pool3d(inputs,
                               ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1],
                               padding=padding,
                               name=sc.name)
    return outputs





def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
  """ Batch normalization on convolutional maps and beyond...
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  
  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
  Return:
      normed:        batch-normalized maps
  """
  with tf.variable_scope(scope) as sc:
    num_channels = inputs.get_shape()[-1].value
    beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                        name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
    decay = bn_decay if bn_decay is not None else 0.9
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    # Operator that maintains moving averages of variables.
    # Need to set reuse=False, otherwise if reuse, will see moments_1/mean/ExponentialMovingAverage/ does not exist
    # https://github.com/shekkizh/WassersteinGAN.tensorflow/issues/3
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        ema_apply_op = tf.cond(is_training,
                               lambda: ema.apply([batch_mean, batch_var]),
                               lambda: tf.no_op())
    
    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)
    
    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(is_training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
  return normed

def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
  """ Batch normalization on FC data.
  
  Args:
      inputs:      Tensor, 2D BxC input
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,], bn_decay)


def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope):
  """ Batch normalization on 1D convolutional maps.
  
  Args:
      inputs:      Tensor, 3D BLC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1], bn_decay)



  
def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope):
  """ Batch normalization on 2D convolutional maps.
  
  Args:
      inputs:      Tensor, 4D BHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay)


def batch_norm(inputs, is_training, scope, moments_dims, bn_decay):
    with tf.variable_scope(scope) as sc:
        num_channels = inputs.get_shape()[-1].value
        beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
        decay = bn_decay if bn_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        # Operator that maintains moving averages of variables.
        # Need to set reuse=False, otherwise if reuse, will see moments_1/mean/ExponentialMovingAverage/ does not exist
        # https://github.com/shekkizh/WassersteinGAN.tensorflow/issues/3
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            ema_apply_op = tf.cond(is_training,
                                   lambda: ema.apply([batch_mean, batch_var]),
                                   lambda: tf.no_op())

        # Update moving average and return current batch's avg and var.
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # ema.average returns the Variable holding the average of var.
        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
    return normed

def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope):
  """ Batch normalization on 3D convolutional maps.
  
  Args:
      inputs:      Tensor, 5D BDHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1,2,3], bn_decay)


def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
  """ Dropout layer.

  Args:
    inputs: tensor
    is_training: boolean tf.Variable
    scope: string
    keep_prob: float in [0,1]
    noise_shape: list of ints

  Returns:
    tensor variable
  """
  with tf.variable_scope(scope) as sc:
    outputs = tf.cond(is_training,
                      lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                      lambda: inputs)
    return outputs

def pairwise_distance(point_cloud):
    batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])#matrix transpose
    point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)
    point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
    return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose


def knn(adj_matrix, k=20):
    neg_adj = -adj_matrix
    _, nn_idx = tf.nn.top_k(neg_adj, k=k)
    return nn_idx

def centroid_point_extract(inputData, clusterNumber):
    centroidPoint = {}
    for i in range(len(inputData)):
        inputCoor = inputData[i]
        centroidPoint.update({i:inputCoor[:,0:clusterNumber,:]})
    return centroidPoint

def k_nn_graph(adj_mat, k = 3):
    # adj_mat : B N K K
    adj_sorted , adj_sort_ind = tf.nn.top_k(input = adj_mat , k = k, sorted=True)
    adj_thresh = adj_sorted[:,:,:, k - 1] # k-th largest ele

    k_nn_adj_mat = tf.where( tf.less(adj_mat , tf.expand_dims( adj_thresh , axis = -1 ) ),
                        x = tf.zeros_like(adj_mat),
                        y = adj_mat,
                        name = 'k_nn_adj_mat'
                        )
    return k_nn_adj_mat

def k_nn_dist_graph(adj_dist_mat, k = 3,non_directed=True):
    # adj_mat : B N K K
    adj_sorted , adj_sort_ind = tf.nn.top_k(input = -adj_dist_mat , k = k+1, sorted=True,name='sort_adj')#last k
    adj_thresh = -adj_sorted[:,:,:, k] # k-th shortest distance (positive)
    #adj_thresh=tf.tile(tf.reduce_mean(adj_thresh,axis=-1,keep_dims=True),[1,1,adj_dist_mat.shape[2]])
    k_nn_adj_mat = tf.where( tf.less_equal(adj_dist_mat ,tf.expand_dims( adj_thresh , axis = -1,name='thresh' )),
                        x = adj_dist_mat,
                        y = tf.zeros_like(adj_dist_mat),
                        name = 'k_nn_dist_mat'
                        )
    if non_directed:
        k_nn_adj_mat_inv=tf.transpose(k_nn_adj_mat,[0,1,3,2],name='k_nn_adj_mat_inv')
        k_nn_adj_mat_non_d = tf.where(tf.less(k_nn_adj_mat,k_nn_adj_mat_inv),
                                x=k_nn_adj_mat_inv,
                                y=k_nn_adj_mat,
                                name='k_nn_dist_mat_non_d'
                                )
        return k_nn_adj_mat_non_d
    return k_nn_adj_mat

def cal_pairwise_dist(local_cord):
    loc_matmul = tf.matmul(local_cord , local_cord, transpose_b=True)
    loc_norm = local_cord * local_cord # B N K m
    r = tf.reduce_sum(loc_norm , -1, keep_dims = True) # B N K 1
    r_t = tf.transpose(r, [0,1,3,2]) # B N 1 K
    D = tf.abs(r - 2*loc_matmul + r_t,name='adj_D')
    return D

def adj_mat_euclidean(local_cord,D=None ,normalized = True,self_connected=False,k=None):
    in_shape = local_cord.get_shape().as_list()
    if D==None:
        D = cal_pairwise_dist(local_cord)
    else:
        D = tf.expand_dims(D,axis=1)
    D = tf.matrix_set_diag(D, tf.zeros([in_shape[0], in_shape[1], in_shape[2]]))  # fix Precision problem
    #calculate pairwise distance first

    if k!=None:
        D = k_nn_dist_graph(D,k=k) #get knn dist graph

    if normalized:
        sqrt_D=tf.sqrt(D,name='sqrt_D')
        D_max = tf.reduce_max( tf.reshape(sqrt_D , [in_shape[0] , in_shape[1] , in_shape[2] ,in_shape[2]]) , axis = -1,name='max_row')
        D_max = tf.square(tf.reduce_mean(D_max,axis=-1),name='Dmax_square')
        D_max = tf.expand_dims(tf.expand_dims(D_max , -1), -1)
        D_max = tf.tile(D_max , [1,1,in_shape[2],in_shape[2]])
        D = tf.divide(D , D_max,'normalized_D')

    adj_mat = tf.where( tf.not_equal(D,0),
                        x = tf.exp(-2*D),
                        y = tf.zeros_like(D),
                        name = 'adj_mat'
                        )#this operation would set dialog to zero
    if self_connected:
        adj_mat = tf.matrix_set_diag(adj_mat, tf.ones([in_shape[0], in_shape[1], in_shape[2]]))
    return adj_mat

def tf_normalized_L(pc,k=None,rescale=False,pw_dist=None,L_type='normalized'):
    shape=pc.get_shape().as_list()
    pc = tf.reshape(pc, (shape[0], 1, shape[1], shape[2]))
    adj_mat = adj_mat_euclidean(pc,D=pw_dist,normalized=True,self_connected=False,k=k)
    D = tf.reduce_sum(adj_mat, axis=-1,name='D_sum')
    I = tf.ones_like(D, dtype=tf.float32)
    I = tf.matrix_diag(I)
    if L_type=='simple':
        D_tiled=tf.tile(tf.expand_dims(D,axis=-1),[1,1,1,shape[1]])
        adj_mat_normalized=tf.divide(adj_mat,D_tiled)
        L=I-adj_mat_normalized
        L = tf.reshape(L, [shape[0], shape[1], shape[1]], name='simple_L')
    elif L_type=='normalized':
        D_sqrt = tf.divide(1.0, tf.sqrt(D))
        D_sqrt = tf.matrix_diag(D_sqrt)
        normalize_term=tf.matmul(D_sqrt, tf.matmul(adj_mat, D_sqrt),name='normalize_term')
        L =I- normalize_term #normalized laplacian
        L = tf.reshape(L,[shape[0],shape[1],shape[1]],name='normalized_L')
        if rescale==True:
            eigen_value=tf.expand_dims(tf.expand_dims(tf.self_adjoint_eigvals(L)[:,-1],-1),-1)
            eigen_value=tf.tile(eigen_value,[1,shape[1],shape[1]],name='max_eigen')
            L=tf.subtract(tf.divide(2*L,eigen_value),tf.reshape(I,L.shape),'rescaled_L')

    return L

def tf_cluster_index(input,scaledLaplacian,pointNumber,clusterNumber,geometry_variance):#TODO: speed up
    ## brief: first we need to choose the central point, then choose points around the central point
    shape = input.get_shape().as_list()
    batch_size, num_points, num_dims = shape[0], shape[1], shape[2]
    input = tf.reshape(input, (shape[0], 1, shape[1], shape[2]))
    adj_mat = adj_mat_euclidean(input,normalized=False)
    adj_mat=tf.reshape(adj_mat,(shape[0],  shape[1], shape[1]))
    adj_mat=tf.matrix_set_diag(adj_mat,tf.ones([shape[0],shape[1]]))
    #calculate feature space distance
    fs_dist=adj_mat
    # fs_dist = tf.where(tf.not_equal(scaledLaplacian,0),#只关注近邻点
    #                             x=adj_mat,
    #                             y=scaledLaplacian,
    #                             name='fs_dist'
    #                             )
    ########### select the central points left in Coarsening operation ##############
    # 方案一 在原始图求点到临近点特征空间上的距离和，这个距离大说明有显著不同，携带很大信息量，需要保留。
    # fs_dist_min=tf.reduce_min(fs_dist,axis=-1,name='fs_dist_min')
    # _,nn_idx=tf.nn.top_k(-fs_dist_min,k=pointNumber,name='cluster_idx')

    # 方案二 输入特征值和比较大的点比较容易从maxpooling中保留，于是抽取这些点作为下一层的点
    # input = tf.reshape(input, (shape[0], shape[1], shape[2]))
    # input_sum = tf.reduce_sum(input, axis=-1)
    # _, nn_idx = tf.nn.top_k(input_sum, k=pointNumber, name='cluster_idx')

    # 方案三 farthest point sampling 原始数据已经被这样采过,所以抽取前n个点就行
    nn_idx = tf.tile(tf.expand_dims(tf.range(0, pointNumber), 0), [batch_size, 1])
    #nn_idx = tf.tile(tf.expand_dims(tf.random_shuffle(tf.range(0, num_points))[0:pointNumber], 0), [batch_size, 1])
    # 方案4 保留局部几何变化高频区

    # initial = tf.truncated_normal(shape=[geometry_variance.get_shape().as_list()[-1],1], mean=0.5, stddev=0.05)
    # weight=tf.Variable(initial)
    # geometry_variance=tf.reshape(geometry_variance, [-1, geometry_variance.shape[-1]])
    # pooling_weight=tf.matmul(geometry_variance,weight)
    # pooling_weight = tf.reshape(pooling_weight, [batch_size, num_points])
    #pooling_weight=tf.reduce_sum(geometry_variance,axis=-1)
    #_, nn_idx = tf.nn.top_k(geometry_variance, k=num_points, name='cluster_idx')

    ################## select the points around the central points ####################
    idx_=tf.range(batch_size)
    idx_=tf.tile(tf.reshape(idx_,[batch_size,1]),[1,pointNumber])
    # #idx1=tf.concat([tf.expand_dims(idx_,-1),tf.expand_dims(nn_idx,-1)],axis=-1)

    # idx_1 = tf.tile(tf.expand_dims(tf.range(0, num_points,delta=tf.cast(num_points/pointNumber,tf.int32)), 0), [batch_size, 1])
    # idx_1 = tf.cast(idx_1, tf.int32)
    # idx_1 = tf.concat([tf.expand_dims(idx_, -1), tf.expand_dims(idx_1, -1)], axis=-1)
    # nn_idx=tf.gather_nd(nn_idx,idx_1)

    idx1 = tf.concat([tf.expand_dims(idx_, -1), tf.expand_dims(nn_idx, -1)], axis=-1)
    select_dist=tf.gather_nd(fs_dist,idx1)
    _,idx2=tf.nn.top_k(select_dist,k=clusterNumber) #在feature space中挑选离中心点最近的k个点




    return idx1,idx2