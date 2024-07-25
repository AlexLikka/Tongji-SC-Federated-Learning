import scipy.io
import torch
import torchvision
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import loadmat
import torch.utils.data as data
import numpy as np

torch.set_default_dtype(torch.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

user_count = 6
c_epochs = 100  # central epochs
c_rate = 0.5  # central learning rate

torch.manual_seed(2)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(2)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(6, 20, kernel_size=(20, 2))
        self.conv2 = nn.Conv2d(20, 50, kernel_size=(20, 2))
        self.conv3 = nn.Conv2d(50, 100, kernel_size=(20, 2))
        self.conv4 = nn.ConvTranspose2d(100, 50, kernel_size=(20, 2))
        self.conv5 = nn.ConvTranspose2d(50, 20, kernel_size=(20, 2))
        self.conv6 = nn.ConvTranspose2d(20, 6, kernel_size=(20, 2))

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
snr_data = train_data['snr_data']  # Assuming snr_data is added in Train3000.mat

input_data = torch.tensor(input_data, dtype=torch.float64).to(device)
label = torch.tensor(label, dtype=torch.float64).to(device)
snr_data = torch.tensor(snr_data, dtype=torch.float64).unsqueeze(1).unsqueeze(2).to(device)  # Adding dimensions

# Concatenating SNR data to input data
input_data = torch.cat((input_data, snr_data.repeat(1, input_data.size(1), input_data.size(2), 1)), dim=3)

permuted_input_data = input_data.permute(3, 2, 0, 1).to(device)
permuted_label = label.permute(3, 2, 0, 1).to(device)

# 加载多个测试数据集(仅供参考)
test_datasets = []
for snr_value in range(0, 10, 2):
    test_data = loadmat(f'Test600_{snr_value}.mat', mat_dtype=True)
    test_input_data = test_data['input_data']
    test_label = test_data['output_data']
    snr_data_test = torch.tensor([snr_value] * test_input_data.shape[3], dtype=torch.float64).unsqueeze(1).unsqueeze(2).to(device)
    
    test_input_data = torch.tensor(test_input_data, dtype=torch.float64).to(device)
    test_label = torch.tensor(test_label, dtype=torch.float64).to(device)
    
    # Concatenating SNR data to test input data
    test_input_data = torch.cat((test_input_data, snr_data_test.repeat(1, test_input_data.size(1), test_input_data.size(2), 1)), dim=3)
    
    permuted_test_input_data = test_input_data.permute(3, 2, 0, 1).to(device)
    permuted_test_label = test_label.permute(3, 2, 0, 1).to(device)
    
    test_datasets.append((permuted_test_input_data, permuted_test_label))

class MyDataset(data.Dataset):
    def __init__(self, x, y):
        self.x = x.to(device)
        self.y = y.to(device)
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


train_dataset = MyDataset(permuted_input_data, permuted_label)
dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset) // user_count,
                                         shuffle=True)

for batch_idx, (data, target) in enumerate(dataloader):
    print('Preparing data for batch', batch_idx)
    user_data.append(data.to(device))
    user_target.append(target.to(device))
print('Data loaded')

network_list = []
for user in range(user_count):
    network_list.append(Net().to(device))


def combine_models(models):
    combined_model = Net().to(device)
    combined_params = list(combined_model.parameters())
    param_lists = [[] for _ in range(len(combined_params))]
    for model in models:
        for idx, param in enumerate(model.parameters()):
            param_lists[idx].append(param.data.to(device))
    with torch.no_grad():
        for combined_param, param_list in zip(combined_params, param_lists):
            stacked_params = torch.stack(param_list)
            combined_param.copy_(stacked_params.mean(dim=0))

    return combined_model


for user in range(user_count):
    for i in range(5):
        print(f'Epoch for {user} {i}')
        user_data[user], user_target[user] = user_data[user].to(device), user_target[user].to(device)
        train(network_list[user], user_data[user], user_target[user])
        test_loss = test(network_list[user], user_data[user], user_target[user])

cn = combine_models(network_list)

data_each = []
for snr_value in range(0, 10, 2):
    test_data = loadmat(f'Test600_{snr_value}.mat', mat_dtype=True)
    test_input_data = test_data['input_data']
    test_label = test_data['output_data']
    snr_data_test = torch.tensor([snr_value] * test_input_data.shape[3], dtype=torch.float64).unsqueeze(1).unsqueeze(2).to(device)
    
    test_input_data = torch.tensor(test_input_data, dtype=torch.float64).to(device)
    test_label = torch.tensor(test_label, dtype=torch.float64).to(device)
    test_input_data = torch.cat((test_input_data, snr_data_test.repeat(1, test_input_data.size(1), test_input_data.size(2), 1)), dim=3)
    
    permuted_test_input_data = test_input_data.permute(3, 2, 0, 1).to(device)
    permuted_test_label = test_label.permute(3, 2, 0, 1).to(device)
    test_losses = []
    for c_epoch in range(c_epochs):
        print(f'Epoch for cn {c_epoch}')
        chosen_user = random.choice(range(user_count))
        user_data[chosen_user], user_target[chosen_user] = user_data[chosen_user].to(device), user_target[chosen_user].to(device)
        test_loss = test(cn, user_data[chosen_user], user_target[chosen_user])
        test_losses.append(test_loss)
    ave = np.mean(test_losses)
    data_each.append(ave)

# Test with all test datasets
#for snr_idx, (test_input, test_label) in enumerate(test_datasets):
    #test_loss = test(cn, test_input, test_label)
    #print(f'Test loss for SNR {snr_idx*2}: {test_loss}')

# 保存结果到文件
with open('results.txt', 'w') as f:
    for ave in data_each:
        f.write(str(ave) + '\n')
