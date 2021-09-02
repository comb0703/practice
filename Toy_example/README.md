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

* i did 2, 4, 8 num_workers test, when num_workers is getting bigger, i dont know why, pigure looks like a more having intra compactness (it should be same)


## Performance
* num_workers 2

![worker2](https://user-images.githubusercontent.com/87002037/131765401-e2f8198d-ec96-4862-a1e7-e91bd9dc2385.gif)

* num_workers 4

![worker4](https://user-images.githubusercontent.com/87002037/131765404-ac946c8f-c4a7-4b43-bf04-335424420a32.gif)

* num_workers 8

![worker8](https://user-images.githubusercontent.com/87002037/131765408-c8703e04-5027-4e03-b270-6e29137f8ec7.gif)





* learning 0.1(15 epochs)

![learning 0 1](https://user-images.githubusercontent.com/87002037/131765362-d3fd1168-625c-497b-ac0c-7b9031f85a6b.gif)

* learning 0.01(100 epochs)

![learning_0 01](https://user-images.githubusercontent.com/87002037/131765373-1c1574e8-f874-4e69-9f22-667f658a542c.gif)

## Reference

* A Discriminative Feature Learning Approach for Deep Face Recognition (Yandong WenKaipeng ZhangZhifeng Li)
 
 (ECCV 2016: Computer Vision – ECCV 2016 pp 499-515)

