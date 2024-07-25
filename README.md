# cnncpp

Reference implementation of a CNN in cpp. This is my pet project to get a
better understanding of what happens under the hood of Convolutional Neural Networks,
and to learn and practice newer standards and algorithms in C++.
Initial goal is to implement a capability to run inference on a MNIST
database and down the road to add support for additional
(at this stage) architechtures if classification networks.

## Tasks

- [x] Layers:
  - [x]  Convolution 2D
    - [x] Implement Convolutions
    - [x] Support for stride > 1
    - [x] Add activation and bias
    - [ ] Add support for padding
  - [x] Flatten
  - [x] - Avg Pooling
  - [x] - Max Pooling
  - [x] - Fully Connected
- [x] Find (and train) a reference implementation of MNIST
  - [x] - Find a way to export weights and load them into cnncpp
- [x] Contruct and run the network (LeNet Arch)

## Progress

First version is compatible with LeNet network trained on MNIST database.
It supports loading hd5 weights exported by keras model and performing an inference.

Current implementation does not support padding.
