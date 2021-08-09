# ArcFace

![version](https://img.shields.io/badge/CUDA-11.1-brightgreen) ![version](https://img.shields.io/badge/cuDNN-8.1.0-blue) ![version](https://img.shields.io/badge/pytorch-1.9.0-orange)



## Main idea
![1](https://user-images.githubusercontent.com/87002037/128664968-c7c7973d-af17-4c10-9daa-4e6b0f55a103.PNG)



## Details

* I implemented part 2.1 Figure 3(toy example) Experiments using MNIST dataset

* I didnt apply scale on arcface loss

* It was quite confusing "where do i have to put ArcFace loss function in network architecture?"

* Most important thing is loss is calculated with logit's output and ground-truth label

* ArcFace is technicque for changing logit's input with margin

* ArcFace loss just add margin to groud truth's weight. people called this as ArcFace loss, so i thought that might be standard for calculating for loss. but it wasnt. Still loss is calculated by Cross-Entropy loss function   

![2](https://user-images.githubusercontent.com/87002037/128664977-b0b250c1-d23f-4e1d-841f-ec85869c5250.PNG)


## Performance
* Softmax

![1](https://user-images.githubusercontent.com/87002037/128678740-424c2325-221b-4895-a8ed-258ed42d9231.PNG)

* ArcFace

![2](https://user-images.githubusercontent.com/87002037/128678762-48a60c7b-696d-440a-87fb-44664a327a56.PNG)

* 3D Visualization

![4](https://user-images.githubusercontent.com/87002037/128678829-ce71e0ff-744d-4faf-b9da-be27c92c2d02.PNG)


## Reference

* ArcFace: Additive Angular Margin Loss for Deep Face Recognition (Jiankang Deng, Jia Guo, Niannan Xue, Stefanos Zafeiriou)

