#%%
import torch
import torch.nn as nn

#%%
# Create a tensor
x = torch.tensor([1, 2, 3])
print(x)


#%%
# create a two layber network to classify cifar10
# input layer: 32x32x3
# hidden layer: 100
# output layer: 10
class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred
    

# %%
# load cifar10 data
from torchvision import datasets, transforms
import torch.utils.data as data_utils
dataset = datasets.CIFAR10(root='./data', train=True, download=True)
# flattern the data
dataset.data = dataset.data.reshape(-1, 32*32*3)
# normalize the data
dataset.data = dataset.data / 255.0
# split dataset into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = data_utils.random_split(dataset, [train_size, val_size])
# create a two layber network to classify cifar10
# input layer: 32x32x3
# hidden layer: 100
# output layer: 10
N, D_in, H, D_out = 64, 32*32*3, 100, 10
# training data ftom dataset
x = torch.tensor(train_dataset.dataset.data, dtype=torch.float)
y = torch.tensor(train_dataset.dataset.targets, dtype=torch.long)

#%%
# create a two layber network to classify cifar10
model = TwoLayerNet(D_in, H, D_out)
# loss function
loss_fn = nn.CrossEntropyLoss()
# optimizer
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# training
for t in range(100):

    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
# %%
# validation with validation dataset
x = torch.tensor(val_dataset.dataset.data, dtype=torch.float)
y = torch.tensor(val_dataset.dataset.targets, dtype=torch.long)
y_pred = model(x)

#report accuracy
correct = 0
for i in range(len(y)):
    if y[i] == y_pred[i].argmax():
        correct += 1
print('Accuracy: ', correct / len(y))

# %%
# show the first 10 images and their predicted labels as text

import matplotlib.pyplot as plt
import numpy as np
x = torch.tensor(val_dataset.dataset.data, dtype=torch.float)
y = torch.tensor(val_dataset.dataset.targets, dtype=torch.long)
y_pred = model(x)
for i in range(10):
    plt.imshow(np.reshape(x[i], (32, 32, 3)))
    plt.show()
    print('Predicted: ', y_pred[i].argmax())
    print('Actual: ', y[i])
    

# %%
