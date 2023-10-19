import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

# Define the neural network structure
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # First convolutional layer. Takes in 3 channels (RGB) and outputs 6 channels. Uses a 5x5 kernel.
        self.conv1 = nn.Conv2d(3, 6, 5)
        
        # Pooling layer with 2x2 window and stride of 2.
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Second convolutional layer. Takes in 6 channels and outputs 16. Uses a 5x5 kernel.
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 10)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Apply first convolution, then ReLU, then pooling
        x = self.pool1(F.relu(self.conv1(x)))
        # Apply second convolution, then ReLU, then pooling
        x = self.pool2(F.relu(self.conv2(x)))
        # Flatten the tensor
        x = torch.flatten(x, 1)
        # Apply three fully connected layers with ReLU activation for the first two
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

def imshow(img, ground_truth_labels, predicted_labels, classes, batch_size):
    img = img / 2 + 0.5
    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    label_str = "   Original Value: "
    for i in range(batch_size):
        label_str += f'{classes[ground_truth_labels[i]]} | '

    label_str += "\nPredicted Value: "

    for j in range(batch_size):
        label_str += f'{classes[predicted_labels[j]]} | '

    plt.annotate(label_str, (-0.09, -0.8), xycoords="axes fraction",color='black', fontsize=10, weight='bold',
                 backgroundcolor='white')

    plt.yticks([])
    plt.xticks([])

    plt.show()

def visualize_kernels(kernels, axs, num_kernels=6):
    """Visualizes the first `num_kernels` kernels in the given kernels."""
    """GPT GENERATED"""
    kernels = kernels.detach().cpu().numpy()
    print("Shape of kernels:", kernels.shape)  # Check the shape

    if len(kernels.shape) == 4:  # Only transpose if it's a 4D array
        # Transpose to get dimensions as: [out_channels, in_channels, kernel_height, kernel_width]
        kernels = np.transpose(kernels, [0, 1, 2, 3])
    elif len(kernels.shape) == 3:
        # Expand the 3D array to 4D array with one `in_channel`
        kernels = np.expand_dims(kernels, axis=1)

    count = 0
    for out_channel in range(kernels.shape[0]):
        for in_channel in range(kernels.shape[1]):
            if count >= num_kernels:  # Limit the number of kernels to display
                return
            if count < len(axs):
                kernel = kernels[out_channel, in_channel, :, :]
                axs[count].imshow(kernel, cmap='gray', vmin=-1, vmax=1)  # Specifying vmin and vmax
                axs[count].axis('off')
            count += 1

        
        
def visualize_convolutions(model, testloader):
    """Visualize convolutions for the given model and testloader."""
    """GPT GENERATED"""
    activations = []
    conv_layers = [model.conv1, model.pool1, model.conv2, model.pool2]
    layer_names = ['conv1', 'pool1', 'conv2', 'pool2']

    def hook_fn(module, input, output):
        activations.append(output)

    hooks = []
    for layer in conv_layers:
        hooks.append(layer.register_forward_hook(lambda module, input, output: hook_fn(module, input, output)))


    dataiter = iter(testloader)
    images, labels = next(dataiter)

    activations.clear()
    outputs = model(images)

    for hook in hooks:
        hook.remove()

    fig, axs = plt.subplots(len(conv_layers) + 2, 2, figsize=(10, 25))
    
    # Display the un-normalized input image
    img_unnormalized = torchvision.utils.make_grid(images[0]).numpy()
    img_unnormalized = np.transpose(img_unnormalized, (1, 2, 0))
    img_unnormalized = img_unnormalized / 2 + 0.5  # Un-normalize
    axs[0, 0].imshow(img_unnormalized)
    axs[0, 0].axis('off')
    axs[0, 1].text(0.5, 0.5, 'Un-normalized Input\nShape: {}'.format(images[0].shape), ha='center', va='center')

    # Display the normalized input image
    img_normalized = torchvision.utils.make_grid(images[0]).numpy()
    img_normalized = np.transpose(img_normalized, (1, 2, 0))
    axs[1, 0].imshow(img_normalized)
    axs[1, 0].axis('off')
    axs[1, 1].text(0.5, 0.5, 'Normalized Input\nShape: {}'.format(images[0].shape), ha='center', va='center')

    for i, (activation, name) in enumerate(zip(activations, layer_names)):
        img_grid = activation[0].detach().cpu().numpy()
        img_grid_sep = np.concatenate(
            [np.pad(img, pad_width=((15, 30), (5, 5)), mode='constant', constant_values=1) for img in img_grid], axis=1)

        layer_shape = activation[0].shape  # Get the shape of the activation map
        axs[i + 2, 0].imshow(img_grid_sep, cmap="gray")
        axs[i + 2, 0].axis('off')
        axs[i + 2, 1].text(0.5, 0.5, f'{name}\nShape: {layer_shape}', ha='center', va='center')
        
    for ax in axs[:, 1]:
        ax.axis('off')
    
    # Assuming you still want to visualize kernels in a separate figure
    conv_kernels = [model.conv1.weight.data, model.conv2.weight.data]
    kernel_names = ['conv1_kernels', 'conv2_kernels']

    fig, axs = plt.subplots(2, 7, figsize=(15, 5))

    for i, (kernels, name) in enumerate(zip(conv_kernels, kernel_names)):
        axs[i, 0].set_title(name)
        axs[i, 0].axis('off')
        # Assuming the function visualize_kernels is already defined
        visualize_kernels(kernels[0], axs[i, 1:])

    plt.show()





# Function to register hooks
def register_hooks(layers, model):
    """GPT GENERATED"""
    activations = []

    def hook_fn(module, input, output):
        activations.append(output)

    hooks = []
    for layer in layers:
        hooks.append(layer.register_forward_hook(hook_fn))

    return activations, hooks




def main():
     # Ensure that multiprocess works correctly
    torch.multiprocessing.freeze_support()

    # Define transformations for the input data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 5

    # Load CIFAR-10 training dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Load CIFAR-10 testing dataset
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Classes in the CIFAR-10 dataset
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Instantiate the neural network
    net = Net()
    PATH = './cifar_net.pth'

    try:
        # Try to load a previously trained model
        net.load_state_dict(torch.load(PATH))
    except:
        # If not available, train the model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        # Train for 10 epochs
        for epoch in range(10):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # Get input data and corresponding labels
                inputs, labels = data
                
                # Reset gradients
                optimizer.zero_grad()

                # Forward pass, backward pass, and optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Print training statistics
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print(
                        f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0


        print('Finished Training')
        # Save the trained model
        torch.save(net.state_dict(), PATH)

    # Visualize the convolutions
    visualize_convolutions(net, testloader)

    # Test the neural network
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print(outputs.shape)
    predicted_array = predicted
    labels_array = labels
    images_plot = images

    # Print the true labels and the network's predictions
    print('GroundTruth: ', ' '.join(
        f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                  for j in range(batch_size)))

    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # Evaluate the network's performance on the test set
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # Print overall accuracy and accuracy per class
    print(
        f'Accuracy of the network on the {total} test images: {100 * correct // total} %')
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    imshow(torchvision.utils.make_grid(images_plot), labels_array, predicted_array, classes, batch_size)


if __name__ == '__main__':    
   main()