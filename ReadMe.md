## pytorch-jingwei-master

### Introduction
This is a PyTorch(0.4.1) implementation of U-Net. Currently, we get test mIOU of **0.320** on the [Tianchi Competition: Jingwei](https://tianchi.aliyun.com/competition/entrance/231717/introduction).

### Dependencies
pip install torch==0.4.1 
pip install matplotlib pillow tensorboardX tqdm
(-i https://pypi.tuna.tsinghua.edu.cn/simple)
### Notice
Please look through the code first and change the directory paths to ur own paths. 

### Training

For the **0.320** version,  simply run:

    CUDA_VISIBLE_DEVICES=0,1,2,3 python train_unet.py --backbone resnet --lr 0.007 --workers 16 --epochs 50 --batch-size 16 --gpu-ids 0,1,2,3 --checkname UNetResNet34 --eval-interval 1 --dataset jingwei --model-name UNetResNet34 --pretrained --base-size 600 --crop-size 512 --loss-type focal

See full input arguments via :

> python train_unet.py --help

### Submition

> test_overlap.py

### Tricks might work
> @**zhang qiulin** 
>  - [ ] emsemble on test images
>  - [ ] use all images w or w/o label as trainset

>**@song zeyu**
>  - [1] choose crop images with label as trainset 
>  - [2] decrease dilation of aspp module in deeplabv3 (increase IOU)
>  - [3] train with 1024 * 1024
>  - [4] add u-net like decoders (increase classification acc/ decrease IOU)
>  - [5] combine 2 and 4 get 0.32 test score 


>@**wei xinran**
>  - [ ] test time augumentation 


### Tricks might not work
>@**ding yifeng**
>  - [ ] random cutout (provided in transforms)
>  - [ ] random rotate with small angle (provided in transforms)

>@**song zeyu**
>  - [ ] pretrain resnet/se-resnet/se-resnext from scratch
>  - [ ] deeplab with focal loss
>  - [ ] deeplab with class wights from training set 
>  - [ ] deeplab with decoder from pytorch-deeplab-xception-master
