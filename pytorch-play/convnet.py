#%%
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torchvision import datasets, transforms

#%%
# load cifar10 data
dataset = datasets.CIFAR10(root='./data', train=True, download=True)
#split dataset into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = data_utils.random_split(dataset, [train_size, val_size])

# import tensorflow as tf
# import keras
# import keras.layers as layers


#%%
# classifying cifar10 with convnet defined using tensorflow keras layers
# input layer: 32x32x3
# conv layer: 32x32x3 -> 32x32x16
# conv layer: 32x32x16 -> 32x32x32
# maxpool layer: 32x32x32 -> 16x16x32
# conv layer: 16x16x32 -> 16x16x32
# maxpool layer: 16x16x32 -> 8x8x32
# fc layer: 8x8x32 -> 10
# model = keras.Sequential([
#     layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(32, 32, 3)),
#     layers.Conv2D(32, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(32, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Flatten(),
#     layers.Dense(10)
# ])



# model = nn.Sequential(
#     nn.Conv2d(3, 16, kernel_size=3, padding=1),
#     nn.ReLU(),
#     nn.Conv2d(16, 32, kernel_size=3, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(2, 2),
#     nn.Conv2d(32, 32, kernel_size=3, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(2, 2),
#     nn.Flatten(),
#     nn.Linear(8*8*32, 10)
# )


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8*8*32, 10)
    
    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = nn.ReLU()(self.conv3(x))
        x = x.view(-1, 8*8*32)
        x = self.fc1(x)
        return x

# train convnet
# create a convnet to classify cifar10
model = ConvNet()
# loss function
loss_fn = nn.CrossEntropyLoss()
# optimizer
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#%%

