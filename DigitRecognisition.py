import torch
import torch.nn as nn  # Import neural network module
import torch.nn.functional as F
import torch.optim as optim  # Import optimization module
from torchvision import datasets, transforms  # Import for datasets and transformations
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
# torch.set_default_device('cuda')
# Data preparation
transform = transforms.Compose([
    # transforms.RandomCrop(32),
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=(0.5), std=(0.5)) # Normalize with mean 0.5 and standard deviation 0.5
])
img = cv2.imread("newImg.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.bitwise_not(img)

img = transform(img)
print(img)

# img = [(255 - x) * 1.0 / 255.0 for x in img ]

# img = torch.Tensor(tensor_8).unsqueeze(axis=0)
# tensor = tensor_8*255
# tensor = np.array(tensor, dtype=np.uint8)
# if np.ndim(tensor)>3:
#     assert tensor.shape[0] == 1
#     tensor = tensor[0]
# Image.fromarray(tensor).show()

img2 = cv2.imread("newImg0.png")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2 = cv2.bitwise_not(img2)

img2 = transform(img2)

img3 = cv2.imread("newImg2.png")
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
img3 = cv2.bitwise_not(img3)
img3 = transform(img3)


# Load MNIST datasets, applying the defined transformations

# Model definition
model = nn.Sequential(
    nn.Flatten(),     # Flatten images into a single vector
    nn.Linear(784, 1024),  # Fully connected layer with 128 neurons
    nn.ReLU(),           # ReLU activation for non-linearity
    nn.Linear(1024, 10)   # Output layer with 10 neurons (for 10 classes in MNIST)
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
img = img.to(device)
img2 = img2.to(device)
img3 = img3.to(device)
model.load_state_dict(torch.load('model.pth'))
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Create DataLoaders for efficient training and testing data handling
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)  
# train_loader = train_loader.to(device)
# test_loader = test_loader.to(device)
# prediction = int(torch.max(output.data, 1)[1].numpy())
loss_fn = nn.CrossEntropyLoss()  # Common loss function for classification
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer
count = 0

# Training and Evaluation loop
for epoch in range(1):  # Loop for 5 epochs
    # model.train()  # Set the model to training mode
    # for batch_idx, (data, target) in enumerate(train_loader):  # Iterate over batches of data
    #     data = data.to(device)
    #     target = target.to(device)
    #     optimizer.zero_grad()  # Clear gradients from the previous iteration
    #     output = model(data)  # Forward pass through the model
    #     loss = loss_fn(output, target)  # Calculate the loss 
    #     loss.backward()  # Compute gradients (backpropagation)
    #     optimizer.step()  # Update model parameters

    model.eval()  # Set the model to evaluation mode 
    test_loss = 0
    correct = 0
    with torch.no_grad():  # Disable gradient calculations for efficiency
        for data, target in test_loader:  # Iterate over test data
            data = data.to(device)
            target = target.to(device)
            output = model(data)          
            if count==0: 
                # plt.imshow(data[42][0],cmap='gray')
                # plt.show()
                print('!!!!!!!!!!')
                count+=1
                print(data[0].shape)
            test_loss += loss_fn(output, target).item()  # Accumulate test loss 
            pred = output.argmax(dim=1, keepdim=True)  # Get predicted class
            correct += pred.eq(target.view_as(pred)).sum().item()  # Update correct predictions

    test_loss /= len(test_loader.dataset)  # Calculate average test loss
    print('\nEpoch: {}, Test Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    torch.save(model.state_dict(),'model.pth')
# prediction = F.softmax(output)

output=model(img)
# plt.imshow(img[0],cmap='gray')
# plt.show()
print(output)
print(output.argmax(dim=1, keepdim=True))
output=model(img2)
print(output)
print(output.argmax(dim=1, keepdim=True))
output=model(img3)
print(output)
print(output.argmax(dim=1, keepdim=True))
