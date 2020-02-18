import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable


train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                            train=False,
                            transform=transforms.ToTensor())

batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function # LINEAR
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-Linearity # NON - LINEAR
        self.sigmoid = nn.Sigmoid()
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function # LINEAR
        out = self.fc1(x)
        # Non - linearity function # NON-LINEAR
        out = self.sigmoid(out)
        # Linear function (readout) # LINEAR
        out = self.fc2(out)
        return out

input_dim = 28 * 28
hidden_dim = 100
output_dim = 10

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

criterion = nn.CrossEntropyLoss()

learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(model.parameters())

print(len(list(model.parameters())))

# Hidden Layer Parameters
print(list(model.parameters())[0].size())

# FC1 Bias Parameters
print(list(model.parameters())[1].size())

# FC2 Parameters
print(list(model.parameters())[2].size())

# FC2 Bias Parameters
print(list(model.parameters())[3].size())

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        # 2. Clear gradient buffers
        optimizer.zero_grad()

        # Forward pass to get output/logits
        # outputs: 100x10
        outputs = model(images)

        # labels: 100x1
        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                # Load images to a Torch Variable
                images = Variable(images.view(-1, 28*28))
                # Hidden dimension: 100
                # Can be any number
                # Similar term:
                #   Number of neurons
                #   Number of non-linear activation functions

                # Forward pass only to get logits/output
                outputs = model(images)

                # Get predictions from the maximum value
                # predicted: 100x1
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels 100, it should be written as "labels.size(0)"
                #labels: 100x1
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total
            print("Iteration: {}, Loss:{}. Accuracy: {}".format(iter, loss.item, accuracy))

print("")








