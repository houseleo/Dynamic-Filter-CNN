import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tf_util
from transform_nets import input_transform_net

from layers import TI_gcn,graph_cluster_maxpooling
from TI_util import tf_normalized_L,tf_cluster_index
def placeholder_inputs(batch_size, num_point):
  pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
  return pointclouds_pl, labels_pl

class config():
    def __init__(self):
        self.k=[20,10,10,5]
        self.cluster_nn=[6,6,6,2]
        self.feature_dim=[32,64,128,256]
        self.point_number=[1024,256,64,32]
        self.mlp_dim=[16,32]
        self.FC_dim=[400,200]

def get_dfn_model(point_cloud, is_training, bn_decay=None):
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    para=config()
    adj_matrix = tf_util.pairwise_distance(point_cloud)
    nn_idx = tf_util.knn(adj_matrix, k=para.k[0])
    Laplacian = tf_normalized_L(point_cloud, k=150,pw_dist=adj_matrix, rescale=None, L_type='simple')
    geometry_variance=tf.norm(tf.matmul(Laplacian,point_cloud), axis=-1)
    
    edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=para.k[0],keep_central=False)
    
    #Spatial Transformation
    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(edge_feature, is_training, bn_decay, K=3)


    point_cloud_transformed = tf.matmul(point_cloud, transform)
    adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
    nn_idx = tf_util.knn(adj_matrix, k=para.k[0])
    edge_feature = tf_util.get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=para.k[0])
    
    #Transformation ends
    
    #Layer1
    
    weight = tf_util.conv2d(edge_feature, para.mlp_dim[0], [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='weight1', bn_decay=bn_decay)

    weight = tf_util.conv2d(weight, para.feature_dim[0], [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='weight1_l3', bn_decay=bn_decay,activation_fn=None)

    net=weight
    net = tf.reduce_mean(net, axis=-2, keep_dims=True)
    bias_l1 = tf.truncated_normal(shape=[batch_size,1,1,para.feature_dim[0]], mean=0, stddev=0.05)
    bias_l1=tf.Variable(bias_l1, name='bias_l1')
    bias_l1=tf.tile(bias_l1,[1,para.point_number[0],1,1])
    net=tf.nn.relu(net+bias_l1)
    net1 = tf.reduce_max(net, axis=1, keep_dims=True)
    
    #Maxpooling
    
    with tf.variable_scope('l1_cluster'):
        select_idx_l1,index_l1=tf_cluster_index(point_cloud,Laplacian,
                                        pointNumber=para.point_number[1],clusterNumber=para.cluster_nn[0],geometry_variance=geometry_variance)#,ori_idx=ori_idx)
        net = graph_cluster_maxpooling(index_l1, net, batch_size=batch_size,
                                        M=para.point_number[1], k=para.cluster_nn[0], n=para.feature_dim[0])
        geometry_variance=tf.gather_nd(geometry_variance,select_idx_l1)
        point_cloud=tf.gather_nd(point_cloud,select_idx_l1)

        
    #Layer 2
    
    adj_matrix = tf_util.pairwise_distance(point_cloud)
    nn_idx = tf_util.knn(adj_matrix, k=para.k[1])
    
    edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=para.k[1], keep_central=False)
    
    
    weight = tf_util.conv2d(edge_feature, para.mlp_dim[0], [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='weight2', bn_decay=bn_decay)
    
    weight = tf_util.conv2d(weight, para.feature_dim[0]*para.feature_dim[1], [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='weight2_l3', bn_decay=bn_decay,activation_fn=None)
    
    weight=tf.reshape(tf.expand_dims(weight,-1),[batch_size,para.point_number[1],para.k[1],para.feature_dim[0],para.feature_dim[1]])
    net=tf_util.get_edge_feature(net, nn_idx=nn_idx, k=para.k[1],get_neighboor=True)
    net=tf.expand_dims(net,axis=-2)
    net=tf.matmul(net,weight)
    net = tf.reduce_mean(tf.squeeze(net), axis=-2, keep_dims=True)
    bias_l2 = tf.truncated_normal(shape=[batch_size,1,1,para.feature_dim[1]], mean=0, stddev=0.05)
    bias_l2 = tf.Variable(bias_l2, name='bias_l2')
    bias_l2 = tf.tile(bias_l2, [1, para.point_number[1], 1, 1])
    net=tf.nn.relu(net+bias_l2)
    net2 = tf.reduce_max(net, axis=1, keep_dims=True)
    
    #Maxpooling
    with tf.variable_scope('l2_cluster'):
        select_idx_l1,index_l1=tf_cluster_index(point_cloud,Laplacian,
                                        pointNumber=para.point_number[2],clusterNumber=para.cluster_nn[1],geometry_variance=geometry_variance)#,ori_idx=ori_idx)
        net = graph_cluster_maxpooling(index_l1, net, batch_size=batch_size,
                                        M=para.point_number[2], k=para.cluster_nn[1], n=para.feature_dim[1])
        geometry_variance = tf.gather_nd(geometry_variance, select_idx_l1)
        point_cloud=tf.gather_nd(point_cloud,select_idx_l1)

        
        
    #Layer3
    
    adj_matrix = tf_util.pairwise_distance(point_cloud)
    nn_idx = tf_util.knn(adj_matrix, k=para.k[2])
    
    edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=para.k[2], keep_central=False)
    
    weight = tf_util.conv2d(edge_feature, para.mlp_dim[0], [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='weight3', bn_decay=bn_decay)

    weight = tf_util.conv2d(weight, para.feature_dim[1]*para.feature_dim[2], [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='weight3_l3', bn_decay=bn_decay,activation_fn=None)
    
    weight=tf.reshape(weight,[batch_size,para.point_number[2],para.k[2],para.feature_dim[1],para.feature_dim[2]])
    net = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=para.k[2], get_neighboor=True)
    net = tf.expand_dims(net, axis=-2)
    net = tf.matmul(net, weight)
    net = tf.reduce_mean(tf.squeeze(net), axis=-2, keep_dims=True)
    bias_l3 = tf.truncated_normal(shape=[batch_size,1,1,para.feature_dim[2]], mean=0, stddev=0.05)
    bias_l3 = tf.Variable(bias_l3, name='bias_l3')
    bias_l3 = tf.tile(bias_l3, [1, para.point_number[2], 1, 1])
    net=tf.nn.relu(net+bias_l3)
    net3 = tf.reduce_max(net, axis=1, keep_dims=True)
    
    #Maxpoolinj
    with tf.variable_scope('l3_cluster'):
        select_idx_l1,index_l1=tf_cluster_index(point_cloud,Laplacian,
                                        pointNumber=para.point_number[3],clusterNumber=para.cluster_nn[2],geometry_variance=geometry_variance)#,ori_idx=ori_idx)
        net = graph_cluster_maxpooling(index_l1, net, batch_size=batch_size,
                                        M=para.point_number[3], k=para.cluster_nn[2], n=para.feature_dim[2])
        point_cloud = tf.gather_nd(point_cloud, select_idx_l1)

    #Layer 4

    edge_feature=tf.expand_dims(point_cloud,axis=-2)
    weight = tf_util.conv2d(edge_feature, para.mlp_dim[0], [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='weight4', bn_decay=bn_decay)

    weight = tf_util.conv2d(weight, para.feature_dim[2]*para.feature_dim[3], [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='weight4_l3', bn_decay=bn_decay,activation_fn=None)
    weight = tf.reshape(tf.expand_dims(weight, -1), [batch_size, para.point_number[3],para.feature_dim[2],para.feature_dim[3]])
    net = tf.expand_dims(net, axis=-2)
    net = tf.matmul(net, weight)
    net = tf.reduce_mean(tf.squeeze(net), axis=-2, keep_dims=True)
    bias_l4 = tf.truncated_normal(shape=[batch_size,1,1,para.feature_dim[3]], mean=0, stddev=0.05)
    bias_l4 = tf.tile(bias_l4, [1, para.point_number[3], 1, 1])
    bias_l4 = tf.Variable(bias_l4, name='bias_l4')
    net=tf.nn.relu(net)
    net = tf.reshape(net, [batch_size, -1])
    
    #Fully Connected Layer
    
    net = tf_util.fully_connected(net, para.FC_dim[0], bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, para.FC_dim[1], bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')
    total_parameters=0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print('Total parameters number is {}'.format(total_parameters))
    return net, end_points

#import pointfly as pf

def get_loss(pred, label, end_points):
  """ pred: B*NUM_CLASSES,
      label: B, """
  labels = tf.one_hot(indices=label, depth=40)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
  classify_loss = tf.reduce_mean(loss)
  return classify_loss


if __name__=='__main__':
  batch_size = 2
  num_pt = 124
  pos_dim = 3

  input_feed = np.random.rand(batch_size, num_pt, pos_dim)
  label_feed = np.random.rand(batch_size)
  label_feed[label_feed>=0.5] = 1
  label_feed[label_feed<0.5] = 0
  label_feed = label_feed.astype(np.int32)



  with tf.Graph().as_default():
    input_pl, label_pl = placeholder_inputs(batch_size, num_pt)
    pos, ftr = get_model(input_pl, tf.constant(True))

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      feed_dict = {input_pl: input_feed, label_pl: label_feed}
      res1, res2 = sess.run([pos, ftr], feed_dict=feed_dict)













