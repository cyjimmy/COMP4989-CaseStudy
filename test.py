import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def imshow(img, ground_truth_labels, predicted_labels, classes, batch_size):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    label_str = "GT: "
    for i in range(batch_size):
        label_str += f'{classes[ground_truth_labels[i]]} | '

    label_str += "\nPR: "

    for j in range(batch_size):
        label_str += f'{classes[predicted_labels[j]]} | '

    plt.annotate(label_str, (-0.09, -0.8), xycoords="axes fraction",color='black', fontsize=10, weight='bold',
                 backgroundcolor='white')

    plt.yticks([])
    plt.xticks([])

    plt.show()

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 5

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = Net()
    PATH = './cifar_net.pth'

    try:
        net.load_state_dict(torch.load(PATH))
    except:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(10):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print(
                        f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')
        torch.save(net.state_dict(), PATH)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=2)

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    predicted_array = predicted
    labels_array = labels
    images_plot = images

    print('GroundTruth: ', ' '.join(
        f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                  for j in range(batch_size)))

    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for label, predicted in zip(labels, predicted):
                if label == predicted:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    print(
        f'Accuracy of the network on the {total} test images: {100 * correct // total} %')
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    imshow(torchvision.utils.make_grid(images_plot), labels_array, predicted_array, classes, batch_size)
