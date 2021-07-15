# ResNet

![version](https://img.shields.io/badge/CUDA-11.1-brightgreen) ![version](https://img.shields.io/badge/cuDNN-8.1.0-blue) ![version](https://img.shields.io/badge/pytorch-1.9.0-orange)



## Main idea
![2](https://user-images.githubusercontent.com/87002037/125739793-c376ea3d-2c3c-4e63-9248-91bcbc4c2419.PNG)


## Details
I implemented part 4.2. CIFAR-10 on paper("Deep Residual Learning for Image Recognition")

We trained total 3 ResNet having different depth layers and compared with paper err

## Performance
* Increasing cardinality

|ResNeXt29|1x64d|2x64d|4x64d|
|------|---|---|---|
|Test err|5.42%| 0.00%|0.00%|

* Increasing bottleneck_width

||1x64d|1x128d|1x192d|
|------|---|---|---|
|Test err|-%| -%|-%|

Paper performance
![1](https://user-images.githubusercontent.com/87002037/125739808-8af862ab-e6ad-49dc-9829-066e73d5d1b1.PNG)

## Reference

* Aggregated Residual Transformations for Deep Neural Networks (Saining Xie; Ross Girshick; Piotr Dollár; Zhuowen Tu; Kaiming He) - 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
