import numpy as np
import matplotlib.pyplot as plt

# x = [1, 5, 10, 10, 25, 50, 70, 75, 300 ]
# y = [0, 0, 0, 0, 0, 1, 1, 1, 1]
#
# colors = np.random.randn(len(x))
# plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
# plt.ylabel("Fever")
# plt.xlabel("Temperature")
#
# plt.scatter(x, y, c=colors, alpha=0.5)
# plt.show()

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=False)

test_dataset = dsets.MNIST(root='./data',
                            train=False,
                            transform=transforms.ToTensor())

# len(test_dataset)
# type(test_dataset[0])
# test_dataset[0][0].size()

# show_img = test_dataset[0][0].numpy().reshape(28, 28)
# plt.imshow(show_img, cmap='gray')
# plt.show()

batch_size = 100
n_iters = 3000
num_epoches = n_iters / (len(train_dataset) / batch_size)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)


import collections

print(isinstance(train_loader, collections.Iterable))

img_1 = np.ones([28, 28])
img_2 = np.ones([28, 28])
lst = [img_1, img_2]

for i in lst:
    print(i.shape)


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

print(train_dataset[0][0].size())

input_dim = 28 * 28
outpur_dim = 10


#STEP 4: INSTANTIATE MODEL CLASS
model = LogisticRegressionModel(input_dim, outpur_dim)


#STEP 5: INSTANTIATE LOSS CLASS
criterion = nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(model.parameters())
print(len(list(model.parameters())))


# parameters A
print(list(model.parameters())[0].size())

# parameters B
print(list(model.parameters())[1].size())

iter = 0
for epoch in range(int(num_epoches)):
    for i, (images, labels) in enumerate(train_loader):
        # size: 100 * 784
        images = images.view(-1, 28*28).requires_grad_()
        labels = labels

        optimizer.zero_grad()
        # Forward pass to get output/logits
        # Output: 100 x 10
        # Apply Linear Regression:
        #    Output(100 x 10) =   X(100x784) * A(784x10) + Bias(100x10)
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            correct = 0
            total = 0

            for images, labels in test_loader:
                # Load images to a Torch Variable
                images = images.view(-1, 28*28).requires_grad_()

                # Forward pass to get output/logits
                # Output: 100 x 10
                # Apply Linear Regression:
                # Output(100 x 10) =   X(100x784) * A(784x10) + Bias(100x10)
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                # labels.size = 100 (minibatch)
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item, accuracy))

iter_test = 0
for images, labels in test_loader:
    iter_test += 1
    images = images.view(-1, 28*28).requires_grad_()
    outputs = model(images)
    if iter_test == 1:
        print('OUTPUTS')
        print(outputs)
    _, predicted = torch.max(outputs.data, 1)

iter_test = 0
for images, labels in test_loader:
    iter_test += 1
    images = images.view(-1, 28 * 28).requires_grad_()
    outputs = model(images)
    if iter_test == 1:
        print('OUTPUTS')
        print(outputs.size())
    _, predicted = torch.max(outputs.data, 1)


print("")



















