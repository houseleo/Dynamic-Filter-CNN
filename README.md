# Dynamic-Filter-CNN for Object Classification in Point Clouds

## To be note

this repository is the work of Pan Guanghua from Shanghai Jiaotong University and Prince Wang from the University of California-Berkeley. It is an on-going project under the Brain-Inspired Application Technology Center(BATC).


For more info about BATC, please visit: http://bat.sjtu.edu.cn/


Further information please contact [Prince Wang](https://www.linkedin.com/in/prince-wang-19511717a/)

## Overview

We propose a new neural network for classification of objects in 3D point cloud data. We applied four layers of convolution which dynamically generates convolution kernel based on the edge features it learned from the point clouds. For its capability to dynamically generate kernels, model is named Dynamic Filter CNN.


## Model Architecture




## Implementations

The model is implemented in TensorFlow. 

## Citation

Our work borrows from many other papers in the field of 3D Point Cloud Deep Learning. The works we referred to are as follows:


* Garimella, Mihir, and Prathik Naidu. “Beyond the Pixel Plane: Sensing and Learning in 	3D.” The Gradient, The Gradient, 27 Aug. 2018, thegradient.pub/beyond-the-pixel-	plane-sensing-and-learning-in-3d/.

* Li, Yangyan, et al. "PointCNN: Convolution on X-transformed points." Advances in Neural 	Information Processing Systems. 2018.

* Qi, Charles R., et al. "Pointnet: Deep learning on point sets for 3d classification and 	segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern 	Recognition. 2017.

* Qi, Charles Ruizhongtai, et al. "Pointnet++: Deep hierarchical feature learning on point sets 	in a metric space." Advances in Neural Information Processing Systems. 2017.

* Wang, Yue, et al. "Dynamic graph cnn for learning on point clouds." arXiv preprint 	arXiv:1801.07829 (2018).

* Simonovsky, Martin, and Nikos Komodakis. "Dynamic edge-conditioned filters in 	convolutional neural networks on graphs." Proceedings of the IEEE conference on 	computer vision and pattern recognition. 2017.

```

## Acknowledgement
This code is heavily borrowed from [dgcnn](https://github.com/WangYueFt/dgcnn).
