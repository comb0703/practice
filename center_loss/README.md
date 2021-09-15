# Center_loss

![version](https://img.shields.io/badge/CUDA-11.1-brightgreen) ![version](https://img.shields.io/badge/cuDNN-8.1.0-blue) ![version](https://img.shields.io/badge/pytorch-1.9.0-orange)



## Main idea
![ce](https://user-images.githubusercontent.com/87002037/133391304-368da59b-35f4-495b-9d6a-ec7ec00f9039.PNG)




## Details

* I implemented part 3.2 Figure 3 Experiments using MNIST dataset

* I set 0.1 learning late and 0.1 lamda

* Network architecture is ResNet38

* When epoch over 50,100, learning rate was divided 10

* Total training epoch 120


## Performance
* softmax


![softmax](https://user-images.githubusercontent.com/87002037/133391872-19eedb2a-926c-4fa4-b13a-1687a09985cb.gif)




* center_loss

![center_loss](https://user-images.githubusercontent.com/87002037/133392030-78766a3b-13ee-4454-828e-310c27a582ae.gif)




## Reference

* A Discriminative Feature Learning Approach for Deep Face Recognition (Yandong WenKaipeng ZhangZhifeng Li)
 
 (ECCV 2016: Computer Vision – ECCV 2016 pp 499-515)

