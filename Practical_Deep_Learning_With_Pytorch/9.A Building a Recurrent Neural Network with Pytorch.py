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

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # Hidden dimensions: size 100
        self.hidden_dim = hidden_dim

        # Number of hidden layers: size 1
        self.layer_dim = layer_dim

        # Building your RNN
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, input_dim)

        # size: input_dim=28, hidden_dim=100, layer_dim=1
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        # h0 size:   (layer_dim=1, batch_size=100, hidden_dim=100)

        # Hidden dimension: 100
        # Can be any number
        # Similar term:
        #   Number of neurons
        #   Number of non-linear activation functions

        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        # We need to detach the hidden state to prevent exploding/vanishing gradients
        # This is part of truncated backpropagation through time (BPTT)
        # out, hn = self.rnn(x, h0.detach())
        #       "out" size: 100*28*100
        #       "hn"  size: 1*100*100
        out, hn = self.rnn(x, h0)

        # Index hidden state of last time step
        # out.size() --> 100, 28, 10 (100 batch_size images, 28 time steps, 10 labels)
        # out[:, -1, :] --> 100, 10 --> just want last (28th) time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10

        return out

# input_dim == one row of one image (28 x 28)
input_dim = 28

# Hidden dimension: 100
# Can be any number
# Similar term:
#   Number of neurons
#   Number of non-linear activation functions

hidden_dim = 100
layer_dim = 1
output_dim = 10

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)

criterion = nn.CrossEntropyLoss()

learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


#####################
# Parameters in depth
#####################

print(len(list(model.parameters())))

for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())


seq_dim = 28

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, seq_dim, input_dim))
        labels = Variable(labels)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        # outputs.size() --> 100, 10
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        iter += 1

        if (iter % 500 == 0):

            correct = 0
            total = 0

            for images, labels in test_loader:
                images = Variable(images.view(-1, seq_dim, input_dim))
                #labels = Variable(labels)
                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total
            print('Iter: {}, Loss: {}, Accuracy: {}'.format(iter, loss.item(), accuracy))















