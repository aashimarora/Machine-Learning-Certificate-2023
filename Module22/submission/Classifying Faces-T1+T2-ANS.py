#!/usr/bin/env python
# coding: utf-8

# In[59]:


from torchvision.datasets import ImageFolder
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import os


# In[ ]:


#Apply transform

transform = transforms.Compose([
      # Resize images to 32x32 pixels
    transforms.ToTensor(),    
    transforms.Resize((32, 30),antialias=True),# Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values
])


# In[ ]:


class MitchellExpr(Dataset):
    def __init__(self, data_directory="./faces", transform=None):
        self.data_directory = data_directory
        self.transform = transform
        self.filenames = []
        self.labels = []
        
        expr_map = {'sad':0, 'happy':1, 'angry':2, 'neutral':3}
                
        for dirpath, dirnames, filenames in os.walk(data_directory):
            for f in filenames:
                if not f.startswith('.') and os.path.splitext(f)[1].lower() in ['.pgm']:
                    full_path = os.path.join(dirpath, f)  # Construct full path once
                    self.filenames.append(full_path)  # Store full path
                    expr = f.split('_')[2]
                    self.labels.append(expr_map[expr])
                    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img_path = self.filenames[idx]  
        image = Image.open(img_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# In[ ]:


class Mitchell(Dataset):
    def __init__(self, data_directory="./faces", transform=None):
        self.data_directory = data_directory
        self.transform = transform
        self.filenames = []
        self.labels = []
                
        for dirpath, dirnames, filenames in os.walk(data_directory):
            for f in filenames:
                if not f.startswith('.') and os.path.splitext(f)[1].lower() in ['.pgm']:
                    full_path = os.path.join(dirpath, f)  # Construct full path once
                    self.filenames.append(full_path)  # Store full path

                    if 'mitchell' in f.lower():
                        self.labels.append(1)
                    else:
                        self.labels.append(0) 
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img_path = self.filenames[idx]  
        image = Image.open(img_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# In[ ]:


class Net(nn.Module):
    def __init__(self, num_classes, name=None):
        super(Net, self).__init__()
        if name:
            self.name = name
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        
        # compute the total number of parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(self.name + ': total params:', total_params)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[65]:


def create_checkpoint(model, optimizer):
        
    checkpoint = {
        'epoch': 10,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    # Specify the path where you want to save the checkpoint
    checkpoint_path = f'./{model.name}.ck'

    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(model, checkpoint_path, optimizer):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

def train(net, trainloader, *args, **kwargs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    checkpoint = f'./{net.name}.ck'
    if (os.path.exists(checkpoint)):
        load_checkpoint(net, checkpoint, optimizer)
        return

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 6 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 6))
                running_loss = 0.0
        
        create_checkpoint(net, optimizer)


def test(net, testloader, *args, **kwargs):
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 375 test images: %d %%' % (
    100 * correct / total))
        
def main():
 # Training settings
    parser = argparse.ArgumentParser(description='NN Parsing')
    parser.add_argument('--data', type=str, default='./faces', help='Path to directory containing faces dataset.')
  
    #args = parser.parse_args()
    #data_dir = './faces'
    
    #T1
    data = Mitchell(data_directory='./faces', transform=transform)
    len(data.labels)

    net = Net(name='Mitchell', num_classes=2)
    
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])

    print(len(train_data), len(test_data))
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=3,
                                          shuffle=True, num_workers=0)


    testloader = torch.utils.data.DataLoader(test_data, batch_size=5,
                                          shuffle=True, num_workers=0)
    
    train(net, trainloader)
    test(net, testloader)    
      
    #T2
    data = MitchellExpr(data_directory='./faces', transform=transform)
    print(data.__getitem__(0))
    
    net = Net(name='FaceExpression', num_classes=4)
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])

    print(len(train_data), len(test_data))
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=3,
                                          shuffle=True, num_workers=0)


    testloader = torch.utils.data.DataLoader(test_data, batch_size=5,
                                          shuffle=True, num_workers=0)

    train(net, trainloader)
    test(net, testloader)

        
if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




