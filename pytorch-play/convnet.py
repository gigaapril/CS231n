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
testloader = data_utils.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

#%%
print('torch.cuda.is_available()', torch.cuda.is_available())


#%%
# setup tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

#%%
# classifying cifar10 with resnet50 defined using nn.Module
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, 10)

# Trainig just FC for 20 epochs resulted in 52% test accuracy
# training last 10 layers gave 66% accuracy
# training all layers for 20 eochs gave 82% accuracy
# model unlock training last 10 layers
# for param in model.parameters():
#     param.requires_grad = False

# # i = 0
# # for m in model.modules():
# #     if isinstance(m, nn.Conv2d):
# #         i += 1
# #         if i >= 0:
# #             for param in m.parameters():
# #                 param.requires_grad = True

# for param in model.fc.parameters():
#     param.requires_grad = True


model = model.cuda()

#%%
# train the model
loss_fn = nn.CrossEntropyLoss()
# learning_rate = 0.00001
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# # find the best learning rate using lr finder
# from torch_lr_finder import LRFinder
# lr_finder = LRFinder(model, optimizer, loss_fn, device='cuda')
# lr_finder.range_test(trainloader, val_loader=valloader, end_lr=1, num_iter=50)
# lr_finder.plot() # to inspect the loss-learning rate graph
# lr_finder.reset() # to reset the model and optimizer to their initial state


#%%

learning_rate = 5.43E-04
# learning_rate = 5.43E-03
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# train the model with tensorboard and trainloader report accuracy
epochs = 200
global_step = 0
for epoch in range(epochs):
    model.train()
    for i, (images, labels) in enumerate(trainloader):
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, epochs, i+1, len(trainloader), loss.item()))
            writer.add_scalar('training loss', loss.item(), global_step)

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
        writer.add_scalar('validation accuracy', 100 * correct / total, epoch)
# %%
# save the model
torch.save(model.state_dict(), 'convnet.ckpt')

# %%
# test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testloader:
        images = images.to('cuda')
        labels = labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
# %%
print(len(testset))
print(len(val_dataset))
print(len(train_dataset))
# %%
# convert model to torchscript
model = model.cpu()
example = torch.rand(1, 3, 32, 32)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("convnet.pt")

# %%
