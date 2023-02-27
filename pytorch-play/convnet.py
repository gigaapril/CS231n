#%%
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torchvision import datasets, transforms

#%%
# load cifar10 data with transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
batch_size = 1024
cifar_pwd = './data'
trainset = datasets.CIFAR10(root=cifar_pwd, train=True, download=True, transform=transform)

#split dataset into training and validation
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
train_dataset, val_dataset = data_utils.random_split(trainset, [train_size, val_size])

trainloader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valloader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


# load test dataset with transforms
testset = datasets.CIFAR10(root=cifar_pwd, train=False, download=True, transform=transform)
testloader = data_utils.DataLoader(testset, batch_size=batch_size * 2, shuffle=False, num_workers=0)

#%%
print('torch.cuda.is_available()', torch.cuda.is_available())


#%%
# setup tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

#%%
# classifying cifar10 with convnet defined using nn.Module
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# training data ftom dataset
x = torch.tensor(train_dataset.dataset.data, dtype=torch.float)
x = x.permute(0, 3, 1, 2)
y = torch.tensor(train_dataset.dataset.targets, dtype=torch.long, device='cuda')

# train the model
model = ConvNet()
loss_fn = nn.CrossEntropyLoss()
learning_rate = 2.48E-01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loader = data_utils.DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = data_utils.DataLoader(val_dataset, batch_size=1000, shuffle=True)

# find the best learning rate using lr finder
from torch_lr_finder import LRFinder
lr_finder = LRFinder(model, optimizer, loss_fn, device='cuda')
lr_finder.range_test(train_loader, val_loader=val_loader, end_lr=1, num_iter=50)
lr_finder.plot() # to inspect the loss-learning rate graph
lr_finder.reset() # to reset the model and optimizer to their initial state


#%%

learning_rate = 2.86E-01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train the model with tensorboard and trainloader report accuracy
epochs = 10
for epoch in range(epochs):
    model.train()
    for i, (images, labels) in enumerate(trainloader):
        images = images.to('cuda')
        labels = labels.to('cuda')
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, epochs, i+1, len(trainloader), loss.item()))
            writer.add_scalar('training loss', loss.item())

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valloader:
            images = images.to('cuda')
            labels = labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
        writer.add_scalar('validation accuracy', 100 * correct / total)
# %%
