# LeNet_MNIST

![version](https://img.shields.io/badge/CUDA-11.1-brightgreen) ![version](https://img.shields.io/badge/cuDNN-8.1.0-blue) ![version](https://img.shields.io/badge/pytorch-1.9.0-orange)



## Main idea
![1](https://user-images.githubusercontent.com/87002037/131639522-aebe7a34-b76a-4852-abc7-b3965aedee0d.PNG)




## Details

* I implemented part 3.1 Figure 3(toy example) Experiments using MNIST dataset
![worker8 (5)](https://user-images.githubusercontent.com/87002037/131642371-a652a478-bb1c-4236-be2b-d4adcb96cebd.png)

* I tried 0.1 learning late and 0.01 learning late

* When i tried 0.1 learning late, After 15 epoch, i got nan loss

* When i tried 0.01 learning late, There was no problem at all until epoch 100 (if you want more, possible)

* As we can check from image, 0.01 test's performance is better than 0.1 test

* Suddenly i was curious about num_workers in Dataloader, i did test about this as well   

* i did 2, 4, 8 num_workers test, when num_workers is getting bigger, pigure looks like a more having intra compactness


## Performance
* num_workers

![worker2 (5)](https://user-images.githubusercontent.com/87002037/131642287-8c7707d6-1200-4c6c-8bbb-8494bc00d9b5.png)
![worker4 (5)](https://user-images.githubusercontent.com/87002037/131642295-48505dd8-7233-47d3-9821-fd14a5cebc0f.png)
![Uploading worker8 (5).png…]()


* ArcFace

![2](https://user-images.githubusercontent.com/87002037/128678762-48a60c7b-696d-440a-87fb-44664a327a56.PNG)

* 3D Visualization

![4](https://user-images.githubusercontent.com/87002037/128678829-ce71e0ff-744d-4faf-b9da-be27c92c2d02.PNG)


## Reference

* A Discriminative Feature Learning Approach for Deep Face Recognition (Yandong WenKaipeng ZhangZhifeng Li)
 
 (ECCV 2016: Computer Vision – ECCV 2016 pp 499-515)

