import os
from torchvision import transforms as T
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torch
from torchvision import utils
from PIL import Image
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

class MyDataset(Dataset): 
    def __init__(self, root, transforms=None, train=True):
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('_')[-1]))
        imgs_num = len(imgs)
        if train:
            self.imgs = imgs[:int(0.9 * imgs_num)]
        else:
            self.imgs = imgs[int(0.9 * imgs_num):]
    
        if transforms is None:
 
            if not train:
                self.transforms = T.Compose([
                    T.Resize(200), 
                    T.CenterCrop(200), 
                    T.ToTensor(),
                    T.Normalize([0.502],[0.211])
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(200), 
                    T.CenterCrop(200), 
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize([0.502],[0.211])
                ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,index):
        img_path = self.imgs[index]
        label_dict = {'Cr':0,'In':1,'Pa':2,'PS':3,'RS':4,'Sc':5}
        label = self.imgs[index].split('_')[-2].split('\\')[-1]
        label = label_dict[label]
        
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

train_data = MyDataset("data",train=True)
test_data = MyDataset("data",train=False)

batch_size = 64
learning_rate = 0.0001 
channels = 1 
epochs = 20

train_dataloader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
for batch_datas, batch_labels in train_dataloader:
    print(batch_datas.size(),batch_labels.size())
    break

test_dataloader = DataLoader(test_data,batch_size=batch_size,shuffle=True)
for batch_datas, batch_labels in test_dataloader:
    print(batch_datas.size(),batch_labels.size())
    break

# Commented out IPython magic to ensure Python compatibility.
import torchvision
# %matplotlib inline
def imshow(img):
 img = img / 2 + 0.5 
 npimg = img.numpy()
 plt.imshow(np.transpose(npimg, (1, 2, 0)))
 plt.show()
dataiter = iter(train_dataloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
labels

# Commented out IPython magic to ensure Python compatibility.
import torchvision
# %matplotlib inline
def imshow(img):
 img = img / 2 + 0.5
 npimg = img.numpy()
 plt.imshow(np.transpose(npimg, (1, 2, 0)))
 plt.show()
dataiter = iter(test_dataloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
labels

len(train_data.imgs)

len(test_data.imgs)

for img, label in train_data: 
 print("图像img的形状{},标签label的值{}".format(img.shape, label))
 print("图像数据预处理后：\n",img)
 break

class cnn(torch.nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
 
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
            torch.nn.Linear(in_features=25 * 25 * 256, out_features=512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),  
            torch.nn.Linear(in_features=512, out_features=6),
        )
 
    def forward(self, input):
        output = self.model(input)
        return output

model = cnn()
if torch.cuda.is_available():
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

record = {'Train Loss': [],'Test Loss': [], 'Train Acc': [], 'Test Acc': []}

for epoch in range(1, epochs+1):
    process = tqdm(train_dataloader, unit='step')
    model.train(True)
    train_loss, train_correct = 0, 0
    for step, (train_imgs, labels) in enumerate(process):
 
        if torch.cuda.is_available():  
            train_imgs = train_imgs.cuda()
            labels = labels.cuda()
        model.zero_grad()  
        outputs = model(train_imgs)  
        loss = criterion(outputs, labels) 
        predictions = torch.argmax(outputs, dim=1)  
        correct = torch.sum(predictions == labels)
        accuracy = correct / labels.shape[0]  
        loss.backward()  
        optimizer.step() 
        process.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" %  
                                   (epoch, epochs, loss.item(), accuracy.item()))
 
        train_loss = train_loss + loss
        train_correct = train_correct + correct
 
        if step == len(process) - 1:
            tst_correct, totalLoss = 0, 0
            model.train(False) 
            model.eval() 
            with torch.no_grad():
                for test_imgs, test_labels in test_dataloader:
                    if torch.cuda.is_available():
                        test_imgs = test_imgs.cuda()
                        test_labels = test_labels.cuda()
                    tst_outputs = model(test_imgs)
                    tst_loss = criterion(tst_outputs, test_labels)
                    predictions = torch.argmax(tst_outputs, dim=1)
 
                    totalLoss += tst_loss
                    tst_correct += torch.sum(predictions == test_labels)

                train_accuracy = train_correct / len(train_data.imgs)
                train_loss = train_loss / len(train_data.imgs)

                test_accuracy = tst_correct / len(test_data.imgs) 
                test_loss = totalLoss / len(test_data.imgs)
 
                record['Train Loss'].append(train_loss.item()) 
                record['Train Acc'].append(train_accuracy.item())
                record['Test Loss'].append(test_loss.item())
                record['Test Acc'].append(test_accuracy.item())
 
                process.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
                                           (epoch, epochs, train_loss.item(), train_accuracy.item(), test_loss.item(),
                                            test_accuracy.item()))
    process.close()

from matplotlib.ticker import MaxNLocator
plt.plot(record['Test Acc'], color='red', label='Test Acc')
plt.plot(record['Train Acc'], label='Train Acc')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch')
plt.xlim([0, epoch])
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylabel('Accuracy')
plt.title('Train and Test ACC')
plt.legend(loc='lower right')
plt.savefig('ACC')
plt.show()

plt.plot(record['Test Loss'], color='red', label='Test Loss')
plt.plot(record['Train Loss'], label='Train Loss')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch')
plt.xlim([0, epoch])
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylabel('Loss')
plt.title('Train and Test LOSS')
plt.legend(loc='upper right')
plt.savefig('LOSS')
plt.show()