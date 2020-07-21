# Self-Guided Adaptation

Code for our paper [Self-Guided Adaptation: Progressive Representation Alignment for Domain Adaptive Object Detection](https://arxiv.org/abs/2003.08777)

### Acknowledgment

The implementation is built on the pytorch implementation of [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch), please refer to the original project to set up the environment.

### Prerequisites

* Python 2.7 or 3.6
* Pytorch 0.4.0
* CUDA 8.0 or higher

### Data Preparation

* **PASCAL_VOC 07+12**: Please refer [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) for constructing PASCAL VOC Datasets.
* **WaterColor**: Please refer [Cross Domain Detection ](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets). 
* **Night-time(Detrac-night)**: TBA.
* **Citysscape, FoggyCityscape**: Please refer [Cityscape](https://www.cityscapes-dataset.com/).
* **KITTI**: Please refer [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php?type=residential)

### All codes are written to fit for the Data format of Pascal VOC.

### Pretrained Model

We used ResNet101 pretrained on the ImageNet in our experiments. You can download the model from:

* ResNet101: [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)


### Well-trained Domain Adaptation Object Detection Models

* Cityscape to KITTI(Res101-based): [GoogleDrive](https://drive.google.com/open?id=1WJEOWzaM6T5mBimaQniPxb62ipEoaOz5)
* KITTI to Cityscape(Res101-based): [GoogleDrive](https://drive.google.com/open?id=1436aIDyv8xXf8pjlfUxK6J69nN6aeMRb)
* Cityscape to Foggycityscape(Res101-based): [GoogleDrive](https://drive.google.com/open?id=1XJdJHRLYUi6XxJWkm1MZQwEBEeTtQszS)
* Pascal VOC to WaterColor(Res101-based): [GoogleDrive](https://drive.google.com/open?id=1bDjEkJCjz2DHP90ATUQL5wVwD4Qmq2fF)
* Daytime(Cityscape) to Night-time(Detrac-Night) (Res101-based): [GoogleDrive](https://drive.google.com/open?id=1ZAhVHfI4sP4jotUQfc96qWIyChp2LXK_)

### Train
* Train SGA with Self-guided adversarial loss and hardness loss:
```
 CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net_auto.py \
                    --dataset source_dataset --dataset_t target_dataset --net res101 \
                    --cuda
```

* Train SGA with all components(self-guided adversarial loss, hardness loss, self-guided progressive sampling)
```
 CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net_auto_self_pace.py \
                    --dataset source_dataset --dataset_t target_dataset --net res101 \
                    --cuda
```
