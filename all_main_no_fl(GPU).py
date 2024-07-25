import scipy.io
import torch
import torchvision
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch.utils.data as data
import matplotlib
matplotlib.use("Qt5Agg")
torch.set_default_dtype(torch.float64)
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

user_count = 6
c_epochs = 1000  # central epochs
c_rate = 0.5  # central learning rate

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(6, 20, kernel_size=(20,2))
        self.conv2 = nn.Conv2d(20, 50, kernel_size=(20,2))
        self.conv3 = nn.Conv2d(50, 100, kernel_size=(20, 2))
        self.conv4 = nn.ConvTranspose2d(100, 50, kernel_size=(20, 2))
        self.conv5 = nn.ConvTranspose2d(50, 20, kernel_size=(20, 2))
        self.conv6 = nn.ConvTranspose2d(20, 6, kernel_size=(20,2))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        return x

def train(network, data, target):
    optimizer = torch.optim.SGD(network.parameters(), lr=c_rate)
    for epoch in range(10):  # local epochs
        output = network(data)
        loss = F.mse_loss(output, target)  # assuming MSE loss for simplicity
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(network, data, target):
    with torch.no_grad():
        output = network(data)
        loss = F.mse_loss(output, target)
        print(loss)
    return loss.item()


user_data = []
user_target = []

train_data = loadmat('Train3000.mat', mat_dtype=True)
input_data = train_data['input_data']
label = train_data['output_data']
input_data = torch.tensor(input_data, dtype=torch.float64).to(device)
label = torch.tensor(label, dtype=torch.float64).to(device)
permuted_input_data = input_data.permute(3, 2, 0, 1)
permuted_label = label.permute(3, 2, 0, 1)
test_data = loadmat('Test600.mat', mat_dtype=True)
test_input_data = test_data['input_data']
test_label = test_data['output_data']
test_input_data = torch.tensor(test_input_data, dtype=torch.float64).to(device)
test_label = torch.tensor(test_label, dtype=torch.float64).to(device)
permuted_test_input_data = test_input_data.permute(3, 2, 0, 1)
permuted_test_label = test_label.permute(3, 2, 0, 1)

class MyDataset(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.idx = list()
        for item in x:
            self.idx.append(item)
        pass

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        input_data = self.x[index]
        label = self.y[index]
        return input_data, label
        # return tensor_x.to('cuda'), tensor_y.to('cuda')

train_dataset = MyDataset(permuted_input_data, permuted_label)
dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset) // user_count, shuffle=True)
# for batch_x, batch_y in dataloader:
#     print(batch_x, batch_y)
test_dataset = MyDataset(permuted_test_input_data, permuted_test_label)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=True)

# make sure the data is fully distributed
for batch_idx, (data, target) in enumerate(dataloader):
    print('Preparing data for batch', batch_idx)
    user_data.append(data)
    user_target.append(target)
print('Data loaded')


# Assuming user_data and user_target are predefined datasets for each user
# Here, just as placeholders

# Select one random user model
chosen_user = random.choice(range(user_count))
cn = Net().to(device)  # Initialize the network
optimizer = torch.optim.SGD(cn.parameters(), lr=c_rate)

test_losses = []
for c_epoch in range(c_epochs):
    print(f'Epoch {c_epoch}')
    user_data[chosen_user],user_target[chosen_user] =user_data[chosen_user].to(device), user_target[chosen_user].to(device)
    train(cn, user_data[chosen_user], user_target[chosen_user])
    test_loss = test(cn, user_data[chosen_user], user_target[chosen_user])
    test_losses.append(test_loss)

fig = plt.figure()
plt.plot(test_losses)
plt.show()
