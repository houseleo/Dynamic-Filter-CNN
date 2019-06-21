import tensorflow as tf
import sys
#sys.path.append('./utils')
import TI_util as tf_util

def weightVariables(shape, name):
    initial = tf.truncated_normal(shape=shape, mean=0, stddev=0.05)
    return tf.Variable(initial, name=name)


def chebyshevCoefficient(chebyshevOrder, inputNumber, outputNumber):
    chebyshevWeights = dict()
    with tf.name_scope('chebyshevWeights'):
        for i in range(chebyshevOrder):
            initial = tf.truncated_normal(shape=[inputNumber, outputNumber], mean=0, stddev=0.05)
            chebyshevWeights['w_' + str(i)] = tf.Variable(initial)
    return chebyshevWeights

def cheby_gcn(input, scaledLaplacian, pointNumber, outputFeatureN, chebyshev_order,bn_decay=None,norm=False,relu=True):
    inputFeatureN=int(input.shape[-1])
    if norm==True:
        inputFeatureN=1
    biasWeight = weightVariables([outputFeatureN], name='bias_w')
    chebyshevCoeff = chebyshevCoefficient(chebyshev_order, inputFeatureN, outputFeatureN)
    chebyPoly = []
    cheby_K_Minus_1 = tf.matmul(scaledLaplacian, input)
    cheby_K_Minus_2 = input #ie:x
    chebyPoly.append(cheby_K_Minus_2)
    chebyPoly.append(cheby_K_Minus_1)

    for i in range(2, chebyshev_order):
        chebyK = 2 * tf.matmul(scaledLaplacian, cheby_K_Minus_1) - cheby_K_Minus_2# 递推公式
        chebyPoly.append(chebyK)
        cheby_K_Minus_2 = cheby_K_Minus_1
        cheby_K_Minus_1 = chebyK
        #cheby_K_Minus_2, cheby_K_Minus_1 = cheby_K_Minus_1, chebyK
    chebyOutput = []
    for i in range(chebyshev_order):
        weights = chebyshevCoeff['w_' + str(i)]
        if norm == True:
            # chebyPolyNorm = tf.square(chebyPoly[i])
            chebyPolyNorm = tf.norm(chebyPoly[i], axis=-1, keep_dims=True)
            chebyPolyNorm = tf.reshape(chebyPolyNorm, [-1, 1])
            output = tf.matmul(chebyPolyNorm, weights)
        else:
            chebyPolyReshape = tf.reshape(chebyPoly[i], [-1, inputFeatureN])
            output = tf.matmul(chebyPolyReshape, weights)
        output = tf.reshape(output, [-1, pointNumber, outputFeatureN])
        chebyOutput.append(output)
    gcnOutput = tf.add_n(chebyOutput) + biasWeight
    if bn_decay!=None:
        gcnOutput = batch_norm(gcnOutput, tf.constant(True), 'bn1', [0, 1], bn_decay)
    if relu:
        gcnOutput = tf.nn.relu(gcnOutput)
    return gcnOutput

def mask_gcn(input, scaledLaplacian, pointNumber, outputFeatureN, chebyshev_order,k_hop=2,bn_decay=None,mask=None):
    shape=scaledLaplacian.get_shape()
    inputFeatureN=int(input.shape[-1])
    biasWeight = weightVariables([outputFeatureN], name='bias_w')
    chebyshevCoeff = chebyshevCoefficient(chebyshev_order, inputFeatureN, outputFeatureN)
    chebyPoly = []
    I = tf.ones_like(scaledLaplacian[:,:,0], dtype=tf.float32)
    I = tf.matrix_diag(I,name='I')
    cheby_K_Minus_1 = scaledLaplacian#tf.matmul(scaledLaplacian, input)
    cheby_K_Minus_2 = I#input #ie:x
    chebyPoly.append(tf.matmul(cheby_K_Minus_2,input,name='chebyK_2'))
    chebyPoly.append(tf.matmul(cheby_K_Minus_1,input,name='chebyK_1'))
    if mask==None:
        mask=scaledLaplacian#tf.cast(tf.cast(scaledLaplacian,tf.bool),tf.float32)
    with tf.variable_scope('mask'):
        for i in range(1,k_hop):
            mask=tf.matmul(mask,mask,name='mask'+str(i))
            # mask_sum=tf.reduce_sum(mask,axis=-1,name='mask_sum'+str(i),keep_dims=True)
            # mask+=0*tf.tile(mask_sum,[1,1,shape[2]])
        mask = tf.cast(tf.cast(mask,tf.bool),tf.float32)
    #mask = I
    for i in range(2, chebyshev_order):
        # if i<=k_hop:
        #     chebyK =2*tf.matmul(scaledLaplacian, cheby_K_Minus_1)-cheby_K_Minus_2
        # else:
        #     if i==k_hop+1:
        #         mask=tf.cast(tf.cast(cheby_K_Minus_1,tf.bool),tf.float32)
        #     chebyK = 2*mask*tf.matmul(scaledLaplacian, cheby_K_Minus_1) - cheby_K_Minus_2
        chebyK = 2 * mask * tf.matmul(scaledLaplacian, cheby_K_Minus_1) - cheby_K_Minus_2
        chebyPoly.append(tf.matmul(chebyK,input))
        cheby_K_Minus_2 = cheby_K_Minus_1
        cheby_K_Minus_1 = chebyK
    chebyOutput = []
    for i in range(chebyshev_order):
        weights = chebyshevCoeff['w_' + str(i)]
        chebyPolyReshape = tf.reshape(chebyPoly[i], [-1, inputFeatureN])
        output = tf.matmul(chebyPolyReshape, weights)
        output = tf.reshape(output, [-1, pointNumber, outputFeatureN])
        chebyOutput.append(output)

    gcnOutput = tf.add_n(chebyOutput) + biasWeight
    if bn_decay!=None:
        gcnOutput = tf_util.batch_norm(gcnOutput, tf.constant(True), 'bn1', [0, 1], bn_decay)
    gcnOutput = tf.nn.relu(gcnOutput,name='gcn_output')
    return gcnOutput
# def TI_gcn(input, scaledLaplacian, pointNumber, outputFeatureN, chebyshev_order,bn_decay=None,norm=False):
#     shape=scaledLaplacian.get_shape()
#     inputFeatureN=int(input.shape[-1])
#     if norm==True:
#         inputFeatureN=2
#     biasWeight = weightVariables([outputFeatureN], name='bias_w')
#     initial = tf.truncated_normal(shape=[(chebyshev_order+1)*(chebyshev_order+1), outputFeatureN], mean=0, stddev=0.05)
#     chebyshevCoeff= tf.Variable(initial)
#     #chebyshevCoeff = chebyshevCoefficient(1,chebyshev_order*chebyshev_order, outputFeatureN)
#     chebyK_list = []
#     I = tf.ones_like(scaledLaplacian[:,:,0], dtype=tf.float32)
#     I = tf.matrix_diag(I,name='I')
#     cheby_K_Minus_1 = scaledLaplacian#tf.matmul(scaledLaplacian, input)
#     cheby_K_Minus_2 = I#input #ie:x
#     chebyK_list.append(cheby_K_Minus_2)
#     chebyK_list.append(cheby_K_Minus_1)
#
#     for i in range(2, chebyshev_order):
#         chebyK = 2 * tf.matmul(scaledLaplacian, cheby_K_Minus_1) - cheby_K_Minus_2
#         chebyK_list.append(chebyK)
#         cheby_K_Minus_2 = cheby_K_Minus_1
#         cheby_K_Minus_1 = chebyK
#     chebyOutput = []
#     #chebyPoly=[]
#     for i in range(chebyshev_order):
#         #weights = chebyshevCoeff
#         chebyPoly = tf.expand_dims(tf.matmul(chebyK_list[i], input),axis=-2)
#         if i==0:
#             relative_pos=tf.concat([tf.zeros_like(chebyPoly),chebyPoly],axis=-2)
#         else:
#             relative_pos=tf.concat([relative_pos,chebyPoly],axis=-2)
#     rela_dist_graph=tf_util.cal_pairwise_dist(relative_pos)
#     rela_dist_graph = tf.matrix_band_part(rela_dist_graph, -1, 0)#lower triangle
#     rela_dist_graph=tf.reshape(rela_dist_graph,[-1,(chebyshev_order+1)*(chebyshev_order+1)])
#     output=tf.matmul(rela_dist_graph, chebyshevCoeff)
#     output = tf.reshape(output, [-1, pointNumber, outputFeatureN])
#
#     gcnOutput =output + biasWeight
#     if bn_decay!=None:
#         gcnOutput = tf_util.batch_norm(gcnOutput, tf.constant(True), 'bn1', [0, 1], bn_decay)
#     gcnOutput = tf.nn.relu(gcnOutput,name='gcn_output')
#     return gcnOutput

def TI_gcn(input, scaledLaplacian, pointNumber, outputFeatureN, chebyshev_order,bn_decay=None,norm=False,is_training=None):
    shape=scaledLaplacian.get_shape()
    inputFeatureN=int(input.shape[-1])
    if norm==True:
        inputFeatureN=2
    biasWeight = weightVariables([outputFeatureN], name='bias_w')
    chebyshevCoeff = chebyshevCoefficient(chebyshev_order, inputFeatureN, outputFeatureN)
    chebyK_list = []
    I = tf.ones_like(scaledLaplacian[:,:,0], dtype=tf.float32)
    I = tf.matrix_diag(I,name='I')
    cheby_K_Minus_1 = scaledLaplacian#tf.matmul(scaledLaplacian, input)
    cheby_K_Minus_2 = I#input #ie:x
    chebyK_list.append(cheby_K_Minus_2)
    chebyK_list.append(cheby_K_Minus_1)
    for i in range(chebyshev_order):
        chebyK =2 * tf.matmul(scaledLaplacian, cheby_K_Minus_1) - cheby_K_Minus_2
        chebyK_list.append(chebyK)
        cheby_K_Minus_2 = cheby_K_Minus_1
        cheby_K_Minus_1 = chebyK
    chebyOutput = []
    for i in range(chebyshev_order):
        weights = chebyshevCoeff['w_' + str(i)]
        chebyPoly = tf.matmul(chebyK_list[i], input)
        if norm == True:
            # chebyPolyNorm = tf.square(chebyPoly[i])

            chebyPolyNorm = tf.norm(chebyPoly, axis=-1, keep_dims=True)
            # return chebyPolyNorm,chebyPolyNorm
            chebyPolyNorm_mean=tf.tile(tf.reduce_mean(chebyPolyNorm,keep_dims=True,axis=-2),[1,shape[1].value,1])
            #tf.where(chebyPolyNorm>chebyPolyNorm_mean)
            # elif i>1:
            #     geometry_variance=geometry_variance+tf.squeeze(chebyPolyNorm)
            #chebyPolyNorm=tf.divide(chebyPolyNorm,chebyPolyNorm_mean)
            # chebyPolyNorm_tiled = tf.tile(chebyPolyNorm, [1, 1, input.shape[2]])
            # input_1=tf.divide(chebyPoly,chebyPolyNorm_tiled)
            chebyPoly_1=tf.matmul(chebyK_list[i],chebyPoly)
            chebyPolyNorm_1 = tf.norm(chebyPoly_1, axis=-1, keep_dims=True)
            # chebyPolyNorm_tiled_1 = tf.tile(chebyPolyNorm_1, [1, 1, input.shape[2]])
            # input_2=tf.divide(chebyPoly_1,chebyPolyNorm_tiled_1)
            # chebyPoly_2=tf.matmul(chebyK_list[i],input_2)
            # chebyPolyNorm_2 = tf.norm(chebyPoly_2, axis=-1, keep_dims=True)
            # if i==1:
            #     geometry_variance=tf.squeeze(chebyPolyNorm)
            # elif i>1:
            #     geometry_variance = tf.concat([geometry_variance,chebyPolyNorm],axis=-1)
            TI_feature=tf.concat([chebyPolyNorm,chebyPolyNorm_1],axis=-1)
            TI_feature = tf.reshape(TI_feature, [-1, TI_feature.shape[-1]])
            output = tf.matmul(TI_feature, weights)
        else:
            chebyPolyReshape = tf.reshape(chebyPoly, [-1, inputFeatureN])
            output = tf.matmul(chebyPolyReshape, weights)

        output = tf.reshape(output, [-1, pointNumber, outputFeatureN])
        chebyOutput.append(output)

    gcnOutput = tf.add_n(chebyOutput) + biasWeight
    if bn_decay!=None:
        gcnOutput = tf_util.batch_norm(gcnOutput, tf.constant(True), 'bn1', [0, 1], bn_decay)
    gcnOutput = tf.nn.relu(gcnOutput,name='gcn_output')
    return gcnOutput#,geometry_variance

def globalPooling(gcnOutput):
    poolingOutput = tf.reduce_max(gcnOutput, axis=[1])
    return poolingOutput

def graph_cluster_maxpooling(batch_index, batch_feature_maps,batch_size,M, k, n, points=None):
    # Description: max pooling on each of the cluster
    # input: (1)index function: B*M*k (batch_index)
    #       (2)feature maps: B*N*n1 (batch_feature_maps)
    #       (3) batch_size
    #       (4) cluster size M
    #       (5) nn size k
    #       (6) n feature maps dimension
    # output: (1)B*M*n1 (after max-pooling on each of the cluster)


    index_reshape = tf.reshape(batch_index, [M*k*batch_size, 1])
    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1))
    batch_idx_tile = tf.tile(batch_idx, (1, M * k))
    batch_idx_tile_reshape = tf.reshape(batch_idx_tile, [M*k*batch_size, 1])
    new_index = tf.concat([batch_idx_tile_reshape, index_reshape], axis=1)
    group_features = tf.gather_nd(batch_feature_maps, new_index)


    group_features_reshape = tf.reshape(group_features, [batch_size, M, k, n])

    max_features = tf.reduce_max(group_features_reshape, axis=2)

    if points!=None:
        ds_points=tf.gather_nd(tf.expand_dims(points,-2), new_index)
        ds_points = tf.reshape(ds_points, [batch_size, M, k, points.shape[-1]])
        ds_points = tf.reduce_mean(ds_points, axis=-2)
        ds_points = tf.reshape(ds_points, [batch_size, M, points.shape[-1]])
        return max_features,ds_points
    else:
        return max_features

#fully connected layer without relu activation
def fullyConnected(features, inputFeatureN, outputFeatureN, relu = True):
    weightFC = weightVariables([inputFeatureN, outputFeatureN], name='weight_fc')
    biasFC = weightVariables([outputFeatureN], name='bias_fc')
    outputFC = tf.matmul(features,weightFC)+biasFC
    if relu==True:
        outputFC=tf.nn.relu(outputFC)
    return outputFC

def input_transform_net(point_cloud, is_training, para,use_bn=False,bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    input_image = tf.expand_dims(point_cloud, -1) #类似在 C×1的图片上卷积
    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=use_bn, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=use_bn, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=use_bn, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [-1, 1024])
    net = tf_util.fully_connected(net, 512, bn=use_bn, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=use_bn, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_XYZ') as sc:
        assert(K==3)
        weights = tf.get_variable('weights', [256, 3*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [3*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [-1, 3, K])
    return transform

def get_edge_feature(point_cloud, nn_idx, k=20):
  """Construct edge feature for each point
  Args:
    point_cloud: (batch_size, num_points, 1, num_dims)
    nn_idx: (batch_size, num_points, k)
    k: int

  Returns:
    edge features: (batch_size, num_points, k, num_dims)
  """
  og_batch_size = point_cloud.get_shape().as_list()[0]
  point_cloud = tf.squeeze(point_cloud)
  if og_batch_size == 1:
    point_cloud = tf.expand_dims(point_cloud, 0)

  point_cloud_central = point_cloud

  point_cloud_shape = point_cloud.get_shape()
  batch_size = point_cloud_shape[0].value
  num_points = point_cloud_shape[1].value
  num_dims = point_cloud_shape[2].value

  idx_ = tf.range(batch_size) * num_points
  idx_ = tf.reshape(idx_, [batch_size, 1, 1])

  point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
  point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)
  point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

  point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

  edge_feature = point_cloud_neighbors-point_cloud_central#tf.concat([point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)
  return edge_feature


def get_static_feature(point_cloud, k,outputFeature):
    adj_matrix = tf_util.pairwise_distance(point_cloud)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    with tf.name_scope('variable'):
        initial = tf.truncated_normal(shape=[k, outputFeature], mean=0, stddev=0.05)
        kernel=tf.Variable(initial,name='edge_cov')

    shape=point_cloud.get_shape().as_list()
    og_batch_size = shape[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_central = point_cloud

    point_cloud_shape = point_cloud.get_shape()
    batch_size = point_cloud_shape[0].value
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)
    point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)
    centroid=tf.reduce_mean(point_cloud_neighbors,axis=-2,keep_dims=True)
    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    #central_dist=tf.norm(point_cloud_neighbors-point_cloud_central,axis=-1)
    centroid_dist=tf.norm(point_cloud_neighbors-centroid,axis=-1,name='centroid_dist')
    #centroid_central_dist=tf.tile(tf.expand_dims(centroid_dist[:,:,0],axis=-1),[1,1,k])

    #edge_feature = tf.concat([central_dist, centroid_dist], axis=-1)

    edge_feature=centroid_dist#/tf.reduce_mean(centroid_dist)#+central_dist
    # edge_feature = tf.matmul(tf.reshape(edge_feature, [-1, k]), kernel)
    edge_feature = tf_util.conv2d(tf.reshape(edge_feature, [batch_size,num_points, k,1]), outputFeature, [1, k],
                                padding='VALID', stride=[1, 1],
                                bn=False, is_training=True,
                                scope='1', bn_decay=None)
    # edge_feature = tf_util.conv2d(edge_feature, 128, [1, 1],
    #                             padding='VALID', stride=[1, 1],
    #                             bn=False, is_training=True,
    #                             scope='2', bn_decay=None)
    # edge_feature = tf_util.conv2d(edge_feature, outputFeature, [1, 1],
    #                             padding='VALID', stride=[1, 1],
    #                             bn=False, is_training=True,
    #                             scope='3', bn_decay=None)
    #edge_feature=tf.nn.relu(edge_feature)
    return tf.reshape(edge_feature,[shape[0],shape[1],outputFeature])


def TI_feature_encoder(point_cloud, k, outputFeature, flag_normalized=False):
    shape = point_cloud.get_shape().as_list()
    batch_size, num_points, num_dims = shape[0], shape[1], shape[2]
    if batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    adj_matrix = tf_util.pairwise_distance(point_cloud)
    nn_idx = tf_util.knn(adj_matrix, k=k)  # 10*(k//10+1))

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    adj_matrix = tf.reshape(adj_matrix, [-1, num_points])
    dist_graph = tf.gather(adj_matrix, nn_idx + idx_, name='b_n_k_n')

    dist_graph = tf.reshape(dist_graph, [-1, 1])  #
    nn_idx = tf.expand_dims(nn_idx, 2)
    nn_idx = tf.tile(nn_idx, [1, 1, k, 1], name='nn_idx')  # B N K K
    idx_ = tf.range(batch_size) * num_points * k * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1, 1])
    idx__ = tf.range(num_points) * k * num_points
    idx__ = tf.reshape(idx__, [1, num_points, 1, 1])
    idx___ = tf.range(k) * num_points
    idx___ = tf.reshape(idx___, [1, 1, k, 1])
    dist_graph = tf.squeeze(tf.gather(dist_graph, nn_idx + idx_ + idx__ + idx___, name='dist_graph'))
    if flag_normalized:
        D_max = tf.reduce_max(tf.reshape(dist_graph, [shape[0], shape[1], k * k]), axis=-1)
        D_max = tf.expand_dims(tf.expand_dims(D_max, -1), -1)
        D_max = tf.tile(D_max, [1, 1, k, k])
        dist_graph = tf.divide(tf.squeeze(dist_graph), D_max + 1e-8)

    dist_graph = tf.reshape(dist_graph, [batch_size, num_points, k * k, 1], name='dist_graph')
    # mlp 64 128 outputFeature
#     dist_graph = tf_util.conv2d(dist_graph, 64, [1, k * k],
#                                 padding='VALID', stride=[1, 1],
#                                 bn=False, is_training=True,
#                                 scope='1', bn_decay=None)
#     dist_graph = tf_util.conv2d(dist_graph, 128, [1, 1],
#                                 padding='VALID', stride=[1, 1],
#                                 bn=False, is_training=True,
#                                 scope='2', bn_decay=None)
    dist_graph = tf_util.conv2d(dist_graph, outputFeature, [1, k*k],
                                padding='VALID', stride=[1, 1],
                                bn=False, is_training=True,
                                scope='4', bn_decay=None)

    return tf.reshape(dist_graph, [batch_size, num_points, outputFeature], name='ti_feature')