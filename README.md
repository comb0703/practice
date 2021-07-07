# ResNet

![version](https://img.shields.io/badge/CUDA-11.1-brightgreen) ![version](https://img.shields.io/badge/cuDNN-8.1.0-blue) ![version](https://img.shields.io/badge/pytorch-1.9.0-orange)



## Main idea
![1311](https://user-images.githubusercontent.com/87002037/124554446-eedb8780-de70-11eb-9c53-c2fcfa799904.png)

## Details
I implemented 4.2. CIFAR-10 part on paper("Deep Residual Learning for Image Recognition")
We trained total 3 ResNet having different depth layers and compared with paper err

## Performance

||Name|Layers|Paper err|Accuracy|
||------|---|---|---|
||ResNet20|20|8.75|8.52|
||ResNet32|32|7.51|7.42|
||ResNet44|44|7.17|6.97|

||ResNet-20|ResNet-32|ResNet-32|
|------|---|---|---|
|Test|91.52% (err 8.48%)|92.54% (err 7.46%)|92.54% (err 7.46%)|
|Paper|91.25% (err 8.75%)|92.49% (err 7.51%)|92.54% (err 7.46%)|

## Reference

* Deep Residual Learning for Image Recognition (Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun; Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778)
