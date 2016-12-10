# Behavioral training

## Design

I started by looking for models similar to my problem, so I can employ transfer learning. The closest thing I knew of for my problem was [deep drive](http://deepdrive.io). However, their data was in Caffe format, so I either had to change from Keras to Caffe or transfer their data to Keras. I thought it would take a lot of time for me to transfer that, and all that time would not really be spent towards the educational goals of this assignment. So I stated looking for something else.

I've tried to reuse the model of deep drive, without reusing the data. They use architecture very similar to AlexNet. However the problem is that Keras does not support merging layers out of the box. I found some AlexNet models for Keras online, but they had issues, presumably due to differences in versions of Keras. I realized I would spend lots of time on this and it would again not really meet my educational goal.

Thern I was trying to reuse the InceptionV3 model with weights for my problem. I've added a fully connected layer at the end and started training. Then I have chosen a portion of neural net to re-train and trained that too. But I found that InceptionV3 seem to overfit no meter what I do - I've even added a few dropout layers with 0.1 coefficients and very strict normalization. I came to a conclusion that this neural network is just too big for my data and I don't have enough time and data to train it.

So I was looking for something smaller then that. I've tried [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), with one modification - I've replaced final fully connected layer of size 10 with fully connected layer of size 1. I have removed softmax activation. And that network seem to fit like a glove. It did not overfit and not too long to train. I've started getting some good results. Image size for this neural net is just 32x32, which is very efficient to store and train. It takes only 3 epochs to train.

I've used GPU at the time of trying InceptionV3, but later I settled for a local CPU. Overhead of uploading training data, downloading weights and managing the GPU machine made it not worth it for working with [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) model.

## Model architecture

This model is inspired by [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) architecture. It has the same top layers, the only difference is the final layers. I have replaced fully connected layer size 10 to a fully connected layer of size 1. I've also removed softmax activation function after that layer.

Model contains of 4 convolution layers and 2 fully connected layres. First pair of convolution layers has "same" and "valid" borders respectivaly, has 32 filters size 3x3, each followed by relu activation function. Above is followed by max pooling 2d of size 2x2. That is followed by dropout layer with probability of 0.25. Let's call above a "Group 1".

"Group 1" is followed by "Group 2", which is same as group one, except each convolution layer now has 64 filters.

"Group 2" is followed by flattern layer and fully connected layer of size 512, "relu" activation function, dropout with probability 0.5 and a fully connected layer of size 1. Last fully connected layer represents a steering angle.

## Creation of training dataset

To generate training data, I've driven through track 3 laps staying in the center of the lane. Then I was practicing recovery by intentionally driving to the edge, and recovering to the center, while only recording the recovery. I've done that for another 3 laps. Then I did same center lane and recovery training on the "off-road" portion of the track. To top it off, I've done some center driving and recovery training while driving on the track in the opposite direction. I've continued by similar training on the second track.

For training I have generated mirrored data of center camera, respectively flipping the steering angle. I've also used left and right cameras subtracting and adding 0.1 coefficient respectively. Finally, I've performed random modifications on the data that would not affect the steering angle, such as width variations, height variations, zoom variations and color shifts.

Not counting random modifications that don't affect steering angle, I ended up with about 44 thousand training samples. Each image was later resized to the 32x32 size in RGB and normalized. This results in about 1 gigabyte of data in memory, which is very manageable on my machine with 8 gigabyte of RAM. Given more time I could optimize copying process of images from one array to another to take less memory.

One other step I could take is to get a joystick and newer version of training application, but I seem to get good enough results without.
