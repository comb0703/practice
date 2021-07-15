# ResNet

![version](https://img.shields.io/badge/CUDA-11.1-brightgreen) ![version](https://img.shields.io/badge/cuDNN-8.1.0-blue) ![version](https://img.shields.io/badge/pytorch-1.9.0-orange)



## Main idea
![1](https://user-images.githubusercontent.com/87002037/125739550-b055f01c-80e6-4021-8822-45e35613f250.PNG)


## Details
I implemented part 4.2. CIFAR-10 on paper("Deep Residual Learning for Image Recognition")

We trained total 3 ResNet having different depth layers and compared with paper err

## Performance


||ResNet-20|ResNet-32|ResNet-44|
|------|---|---|---|
|Test err|10.52%| 9.42%|9.48%|
|Paper err|8.75%|7.51%|7.46%|

## Reference

* Deep Residual Learning for Image Recognition (Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun; Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778)
