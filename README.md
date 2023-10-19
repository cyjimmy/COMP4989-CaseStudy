# COMP4989-CaseStudy

This code repository is an image classifier trained on the CIFAR-10 dataset. It features a simple Convolutional Neural Network (CNN) and demonstrates how to visualize both the convolutional layers and kernels. This project is adapted from the [official PyTorch tutorial on CIFAR-10](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html).

## Features

- Neural network with 2 convolutional layers
- Activation maps visualization
- Kernel visualization
- Training on the CIFAR-10 dataset

## Requirements

- PyTorch
- torchvision
- numpy
- matplotlib

You can install these packages via pip:

```bash
pip install torch torchvision numpy matplotlib
```

## Usage

To train and test the model, simply run the script:

```bash
python main.py
```

## Code Structure

### Neural Network Model (`Net` class)

The network has the following layers:

- First convolutional layer (`conv1`): Takes in an RGB image (3 channels) and outputs 6 channels using a 5x5 kernel.
- Max-pooling layer (`pool1`): 2x2 window with stride of 2.
- Second convolutional layer (`conv2`): Takes in 6 channels and outputs 16 channels using a 5x5 kernel.
- Max-pooling layer (`pool2`): 2x2 window with stride of 2.
- Fully connected layer (`fc1`): Takes 16 * 5 * 5 input features and outputs 10 classes.

### Utility Functions

- `imshow()`: Function to show images with original and predicted labels.
- `visualize_kernels()`: Function to visualize kernels of the convolutional layers.
- `visualize_convolutions()`: Function to visualize activations of convolutional and pooling layers.

### Main Function (`main`)

- Trains the network if no pre-trained model is found.
- Visualizes the convolution layers and kernels.
- Tests the model and prints the accuracy.


## Author

Chun Yip Wong, Simar Vashisht, Sepehr Zohoori Rad

