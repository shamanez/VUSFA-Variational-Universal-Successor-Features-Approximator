# Implementation-Details---VUSFA

VUSFA - https://arxiv.org/abs/1908.06376

This repository contains implementations of USF-A3C Target Driven Visual Navigation. The discription of each model can be found in the following discription.

1. Implementation is the baseline model of Deep Siamese Actor-Critic.

2. Modified Deep Siamese Actor-Critic by replacing pre-trianed Resnet with adding 3 conv layers. 

3. Modified network in the 2nd by adding an LSTM head before the policy and value function prediction heads.

4. Universal Sucessor Feature Dependant Policy (USF-DP) A3C for Target Driven Visual Navigation.

5. Modified network in the 4th by adding a Variational Siamese Bottleneck Layers to improve the tranfer learnign ability.

6. Modified 5th network by only using single frame as the input.


# The AI2Thor simulator can be found in follwing link as a hdf5 data set:

https://drive.google.com/drive/folders/1oUvIZP1GkgCwbIzsRJ-xLgiwfSonQdev?usp=sharing


# All trained models with checkpoints can be found in the following link:

https://drive.google.com/drive/folders/1j06wAvr0fb20BWaC4YWCQDwHbmtUosaP?usp=sharing
