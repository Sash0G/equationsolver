import torch
import torch.nn as nn  # Import neural network module
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim  # Import optimization module
from torchvision import datasets, transforms
import os
import cv2
import numpy as np
import pickle
import random

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(45 * 45, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 17)
    
    def forward(self, x):
        x = x.view(-1, 45 * 45)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5))  # Convert the image to a PyTorch tensor
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# with open('Alltrain.pkl', 'rb') as f:
#     trainMain = pickle.load(f)
# with open('Alltest.pkl', 'rb') as f:
#     testMain = pickle.load(f)
with open('Alltrain2.pkl', 'rb') as f:
    trainMainDiverse = pickle.load(f)
with open('Alltest2.pkl', 'rb') as f:
    testMainDiverse = pickle.load(f)
with open('Alltrain3.pkl', 'rb') as f:
    trainMine = pickle.load(f)
with open('Alltest3.pkl', 'rb') as f:
    testMine = pickle.load(f)

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('allmodelMain2.pth'))
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
# for param in model.parameters():
#     param.requires_grad = False

# # # # Replace the final fully connected layer for fine-tuning
# # # model.fc4 = nn.Linear(model.fc4.in_features, 10)  # Assuming 10 output classes

# # # # Optionally, you can unfreeze specific layers for fine-tuning
# for param in model.fc3.parameters():
#     param.requires_grad = True
# for param in model.fc4.parameters():
#     param.requires_grad = True
model = model.to(device)
batch_size = 64
# while True:
#     train_loader = DataLoader(trainMainDiverse, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(testMainDiverse, batch_size=batch_size, shuffle=False)
#     for epoch in range(10):
#         model.train()
#         for data,target in train_loader:
#             data = data.to(device)
#             target = target.to(device)
#             optimizer.zero_grad()
#             output = model(data)
#             loss = loss_fn(output,target)
#             loss.backward()
#             optimizer.step()
#         model.eval()
#         total_loss = 0.0
#         correct_predictions = 0
#         total_samples = 0
#         with torch.no_grad():  # Disable gradient calculation during evaluation
#             for data, target in test_loader:
#                 data = data.to(device)
#                 target = target.to(device)
#                 outputs = model(data)
#                 loss = loss_fn(outputs, target)
#                 total_loss += loss.item() * data.size(0)
#                 _, predicted = torch.max(outputs, 1)
#                 correct_predictions += (predicted == target).sum().item()
#                 total_samples += target.size(0)
#         average_loss = total_loss / total_samples
#         accuracy = correct_predictions / total_samples*100
#         print('\nEpoch: {}, Test Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,average_loss,correct_predictions,total_samples,accuracy))
#         if accuracy>98.5: break    
#     if accuracy>98.5: break    
# torch.save(model.state_dict(),'allmodelMain.pth')
optimizer = optim.SGD(model.parameters(), lr=0.001)
while True:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc3.parameters():
        param.requires_grad = True
    for param in model.fc4.parameters():
        param.requires_grad = True
    while True:
        train_loader = DataLoader(trainMainDiverse, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(testMainDiverse, batch_size=batch_size, shuffle=False)
        for epoch in range(10):
            model.train()
            for data,target in train_loader:
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output,target)
                loss.backward()
                optimizer.step()
            model.eval()
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            with torch.no_grad():  # Disable gradient calculation during evaluation
                for data, target in test_loader:
                    data = data.to(device)
                    target = target.to(device)
                    outputs = model(data)
                    loss = loss_fn(outputs, target)
                    total_loss += loss.item() * data.size(0)
                    _, predicted = torch.max(outputs, 1)
                    correct_predictions += (predicted == target).sum().item()
                    total_samples += target.size(0)
            average_loss = total_loss / total_samples
            accuracy = correct_predictions / total_samples*100
            print('\nEpoch: {}, Test Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,average_loss,correct_predictions,total_samples,accuracy))
            # torch.save(model.state_dict(),'allmodelMain2.pth')
            if accuracy>98.5: break
        if accuracy>98.5: break

    # for param in model.fc3.parameters():
    #     param.requires_grad = False
    while True:
        train_loader = DataLoader(trainMine, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(testMine, batch_size=batch_size, shuffle=False)
        for epoch in range(10):
            model.train()
            for data,target in train_loader:
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output,target)
                loss.backward()
                optimizer.step()
            model.eval()
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            with torch.no_grad():  # Disable gradient calculation during evaluation
                for data, target in test_loader:
                    data = data.to(device)
                    target = target.to(device)
                    outputs = model(data)
                    loss = loss_fn(outputs, target)
                    total_loss += loss.item() * data.size(0)
                    _, predicted = torch.max(outputs, 1)
                    correct_predictions += (predicted == target).sum().item()
                    total_samples += target.size(0)
            average_loss = total_loss / total_samples
            accuracy = correct_predictions / total_samples*100
            print('\nEpoch: {}, Test Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,average_loss,correct_predictions,total_samples,accuracy))
            # torch.save(model.state_dict(),'allmodelMain2.pth')
            if accuracy>99.5: break
        if accuracy>99.5: break
    test_loader = DataLoader(testMainDiverse, batch_size=batch_size, shuffle=False)
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():  # Disable gradient calculation during evaluation
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            loss = loss_fn(outputs, target)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == target).sum().item()
            total_samples += target.size(0)
    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples*100
    print('\nEpoch: {}, Test Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,average_loss,correct_predictions,total_samples,accuracy))
    torch.save(model.state_dict(),'allmodelMain2.pth')
    if accuracy>99: break


symbols = ['0','1','2','3','4','5','6','7','8','9','-','del','+','=',')','(','X']
directory = r'D:\GitHub\equationsolver\test'
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    img = transform(img).to(device)
    output=model(img)
    print(filename,symbols[output.argmax(dim=1, keepdim=True)[0]])