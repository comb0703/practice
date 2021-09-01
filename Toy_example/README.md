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
* Softmax

![1](https://user-images.githubusercontent.com/87002037/128678740-424c2325-221b-4895-a8ed-258ed42d9231.PNG)

* ArcFace

![2](https://user-images.githubusercontent.com/87002037/128678762-48a60c7b-696d-440a-87fb-44664a327a56.PNG)

* 3D Visualization

![4](https://user-images.githubusercontent.com/87002037/128678829-ce71e0ff-744d-4faf-b9da-be27c92c2d02.PNG)


## Reference

* A Discriminative Feature Learning Approach for Deep Face Recognition (Yandong WenKaipeng ZhangZhifeng Li)
 
 (ECCV 2016: Computer Vision – ECCV 2016 pp 499-515)

