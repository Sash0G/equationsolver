import torch
import torch.nn as nn  # Import neural network module
import torch.nn.functional as F
import torch.optim as optim  # Import optimization module
from torchvision import datasets, transforms
import os
import cv2
import numpy as np
import pickle
import random
from tqdm import tqdm


transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=(0.5), std=(0.5)) # Normalize with mean 0.5 and standard deviation 0.5
])
transform2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(degrees=10),  # Rotate the image by a random angle (-10 to 10 degrees)
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Apply random affine transformations (translation)
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, saturation, and hue
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=(0.5), std=(0.5)) # Normalize with mean 0.5 and standard deviation 0.5
])
symbols = ['0','1','2','3','4','5','6','7','8','9','-','del','+','=',')','(','X']
images = []
# for ind, (k) in tqdm(enumerate(symbols)):
#         dir = os.path.join('D:/GitHub/equationsolver/extracted_images/',k)
#         for filename in tqdm(os.listdir(dir)):
#             f = os.path.join(dir, filename)
#             img = cv2.imread(f)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             img = cv2.bitwise_not(img)
#             img = transform(img)
#             images.append((img,ind))

# random.shuffle(images)
# split_index = len(images) // 2
# list1 = images[:split_index]
# list2 = images[split_index:]
# with open('Alltrain.pkl', 'wb') as f:
#     pickle.dump(list1, f)
# with open('Alltest.pkl', 'wb') as f:
#     pickle.dump(list1, f)

# for ind, (k) in tqdm(enumerate(symbols)):
#     dir = os.path.join('D:/GitHub/equationsolver/extracted_images/',k)
#     for filename in tqdm(os.listdir(dir)):
#         f = os.path.join(dir, filename)
#         img = cv2.imread(f)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img = cv2.bitwise_not(img)
#         img = transform2(img)
#         images.append((img,ind))

# random.shuffle(images)
# split_index = len(images) // 2
# list1 = images[:split_index]
# list2 = images[split_index:]
# with open('Alltrain2.pkl', 'wb') as f:
#     pickle.dump(list1, f)
# with open('Alltest2.pkl', 'wb') as f:
#     pickle.dump(list1, f)

images = []

for ind, (k) in tqdm(enumerate(symbols)):
        dir = os.path.join('D:/GitHub/equationsolver/mine/',k)
        for filename in tqdm(os.listdir(dir)):
            f = os.path.join(dir, filename)
            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.bitwise_not(img)
            img = transform(img)
            images.append((img,ind))

for i in range(250):
    for ind, (k) in tqdm(enumerate(symbols)):
        dir = os.path.join('D:/GitHub/equationsolver/mine/',k)
        for filename in tqdm(os.listdir(dir)):
            f = os.path.join(dir, filename)
            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.bitwise_not(img)
            img = transform2(img)
            images.append((img,ind))

random.shuffle(images)
split_index = len(images) // 2
list1 = images[:split_index]
list2 = images[split_index:]
with open('Alltrain3.pkl', 'wb') as f:
    pickle.dump(list1, f)
with open('Alltest3.pkl', 'wb') as f:
    pickle.dump(list1, f)

# for ind, (k) in tqdm(enumerate(symbols)):
#     dir = os.path.join('D:/GitHub/equationsolver/extracted_images/',k)
#     for filename in tqdm(os.listdir(dir)):
#         f = os.path.join(dir, filename)
#         img = cv2.imread(f)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img = cv2.bitwise_not(img)
#         img = transform2(img)
#         images.append((img,ind))

# random.shuffle(images)
# split_index = len(images) // 2
# list1 = images[:split_index]
# list2 = images[split_index:]

# with open('train2.pkl', 'wb') as f:
#     pickle.dump(list1, f)
# with open('test2.pkl', 'wb') as f:
#     pickle.dump(list1, f)
