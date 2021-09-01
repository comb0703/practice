# LeNet_MNIST

![version](https://img.shields.io/badge/CUDA-11.1-brightgreen) ![version](https://img.shields.io/badge/cuDNN-8.1.0-blue) ![version](https://img.shields.io/badge/pytorch-1.9.0-orange)



## Main idea
![1](https://user-images.githubusercontent.com/87002037/131639522-aebe7a34-b76a-4852-abc7-b3965aedee0d.PNG)




## Details

* I implemented part 3.1 Figure 3(toy example) Experiments using MNIST dataset

* I tried 0.1 learning late and 0.01 learning late

* When i tried 0.1 learning late, After 15 epoch, i got nan loss

* When i tried 0.01 learning late, There was no problem at all until epoch 100 (if you want more, possible)

* As we can check from image, 0.01 test's performance is better than 0.1 test

* Suddenly i was curious about num_workers in Dataloader, i did test about this as well   

* i did 2, 4, 8 num_workers test, when num_workers is getting bigger, pigure looks like a more having intra compactness


## Performance
* num_workers (2,4,8)

![worker2 (5)](https://user-images.githubusercontent.com/87002037/131642754-ccd7ae5f-587b-421a-8774-f19c21637903.png)

  ![worker4 (5)](https://user-images.githubusercontent.com/87002037/131643126-7b8e7d32-92df-4ff3-b0cf-06e5dfc754da.png)

![worker8 (5)](https://user-images.githubusercontent.com/87002037/131642766-78b6d444-e1a4-4de9-8a63-16935a1a56a1.png)




* learning 0.1(15 epochs), 0.01(100 epochs)

![learning_0 1 (15)](https://user-images.githubusercontent.com/87002037/131642846-371357b6-8145-400d-8706-364fc99054f7.png)
![learning_0 01 (100)](https://user-images.githubusercontent.com/87002037/131642856-9ebbdb28-5a39-4d5e-9a06-94a99dbb97f2.png)


## Reference

* A Discriminative Feature Learning Approach for Deep Face Recognition (Yandong WenKaipeng ZhangZhifeng Li)
 
 (ECCV 2016: Computer Vision – ECCV 2016 pp 499-515)

