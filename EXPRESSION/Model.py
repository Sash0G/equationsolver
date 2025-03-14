import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import cv2

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
    
def Recognition(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5))  # Convert the image to a PyTorch tensor
    ])
    model = NeuralNetwork()
    model.load_state_dict(torch.load('EQUATION/allmodelMain2.pth'))
    # img = cv2.bitwise_not(img)
    img = transform(img)
    output=model(img)
    symbols = ['0','1','2','3','4','5','6','7','8','9','-','/','+','=',')','(','*']
    return(symbols[output.argmax(dim=1, keepdim=True)[0]])