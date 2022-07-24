# Facenet_NN2_Semi-hard_Triplet_loss_implementation

Python = 3.7.13 , Tensorflow = 2.8.2

This model is Facenet NN2 embedding model implemented from scratch

![embeddings](https://user-images.githubusercontent.com/93965016/180638922-3e3b2d10-5db9-4fcc-b840-d56c6772dd43.jpg)


# Model
![image](https://user-images.githubusercontent.com/93965016/180639053-77540cd0-0f85-4910-9a5c-c879b76728ad.png)
I made this with Facenet NN2 in Facenet paper and refered to inception v1 model.
Before training by semi-hard triplet loss, i trained this by classification task with softmax. so i added a Dense layer at last place with softmax activation function. After training by the softmax classification task, i deleted the last layer i added (loss value is almost 0.06). and than i trained with semi-hard triplet loss.<br>
I used semi-hard triplet loss that worked out semi-hard condition. the semi hard condition is met when a distance from a ancor to a positive data is less then from the ancor to a negative data. 
# Dataset
I trained this model with 10,000 pictures of MNIST dataset.

# Reference

https://arxiv.org/abs/1503.03832 <br>
https://github.com/omoindrot/tensorflow-triplet-loss <br>
https://github.com/tensorflow/addons/blob/master/tensorflow_addons/losses/triplet.py
