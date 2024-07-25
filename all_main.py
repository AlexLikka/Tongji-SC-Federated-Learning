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
# from torchsummary import summary
torch.set_default_dtype(torch.float64)

user_count = 6
user_fraction = 0.5
c_epochs = 1000  # central epochs
c_rate = 0.5  # central learning rate (relevant if fed_svg == True)
learning_rate = 0.001  # local
momentum = 0.05  # local
fed_sgd = True  # use fed_avg when False
local_epochs = 10 # relevant if fed_svg == False

torch.manual_seed(2)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(2)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(6, 20, kernel_size=(20,2))
        self.conv2 = nn.Conv2d(20, 50, kernel_size=(20,2))
        self.conv3 = nn.Conv2d(50, 100, kernel_size=(20, 2))
        self.conv4 = nn.ConvTranspose2d(100, 50, kernel_size=(20, 2))
        self.conv5 = nn.ConvTranspose2d(50, 20, kernel_size=(20, 2))
        self.conv6 = nn.ConvTranspose2d(20, 6, kernel_size=(20,2))
        # self.fc1 = nn.Linear(1500, 512)
        # self.fc2 = nn.Linear(512, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        return x
        # x = x.view(-1, 1500)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        # return F.log_softmax(x, dim=1)


def train(network, data, target):
    iterations = 1
    l_epochs = 1
    if not fed_sgd:
        data = torch.split(data, 64)  # split into 64 part batches if using fed_avg
        iterations = len(data)
        print("We got " + str(iterations) + " iterations")
        target = torch.split(target, 64)
        l_epochs = local_epochs
    else:
        data = [data]  # wrap data in an array so that data has always the same nesting count
        target = [target]

    for local_epoch in range(l_epochs):
        for i in range(iterations):
            global train_losses
            optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                                  momentum=momentum)
            network.train()
            optimizer.zero_grad()
            output = network(data[i])
            # loss = F.nll_loss(output, target[i])
            loss = F.mse_loss(output, target[i])
            loss.backward()
            if not fed_sgd:
                optimizer.step()
                optimizer.zero_grad()
        train_losses.append(loss)
        # print(f'Local epoch: {local_epoch}, Batch loss: {loss}')
    if fed_sgd:
        # https://discuss.pytorch.org/t/please-help-how-can-copy-the-gradient-from-net-a-to-net-b/41226/5
        return [param[1].grad for param in network.named_parameters()]
    else:
        return network.state_dict()

def test(network):
    network.eval()
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            # test_loss += F.nll_loss(output, target.squeeze(1), size_average=False).item()
            test_loss = F.mse_loss(output, target)
            # pred = output.data.max(1, keepdim=True)[1]
            # correct += pred.eq(target.data.view_as(pred)).sum()
    # test_loss /= len(test_loader.dataset)
    # print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    print('Test set: Avg. loss: {:.4f}\n'.format(test_loss))
    return test_loss

def test_output(network):
    network.eval()
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            print(type(output.numpy()))
    return output.numpy()

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
user_data = []
user_target = []

train_data = loadmat('Train3000.mat', mat_dtype=True)
input_data = train_data['input_data']
label = train_data['output_data']
# print(type(input_data))
# print(label.shape)
input_data = torch.tensor(input_data, dtype=torch.float64)
label = torch.tensor(label, dtype=torch.float64)
permuted_input_data = input_data.permute(3, 2, 0, 1)
permuted_label = label.permute(3, 2, 0, 1)
# print(type(permuted_input_data))
# print(permuted_label.size())
# print(len(permuted_label))


# test_data = loadmat('Train100.mat', mat_dtype=True)
test_data = loadmat('Test600.mat', mat_dtype=True)
test_input_data = test_data['input_data']
test_label = test_data['output_data']
# print(type(test_input_data))
# print(test_label.shape)
test_input_data = torch.tensor(test_input_data, dtype=torch.float64)
test_label = torch.tensor(test_label, dtype=torch.float64)
permuted_test_input_data = test_input_data.permute(3, 2, 0, 1)
permuted_test_label = test_label.permute(3, 2, 0, 1)
# print(type(permuted_test_input_data))
# print(permuted_test_label.size())


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

active_user_count = int(user_count * user_fraction)
network_list = []
train_losses = []
#
# cn = Net().cuda()  # central_network
# summary(cn,(1,64,8))

cn = Net() # central_network
for user in range(user_count):
    network_list.append(Net())

optimizer = torch.optim.SGD(cn.parameters(), lr=c_rate, momentum=0.5)

test_losses = [test(cn)]

for c_epoch in range(c_epochs):
    print(f'GLOBAL EPOCH {c_epoch}')
    chosen_users = random.sample(range(user_count), active_user_count)
    if fed_sgd:
        for user_number in chosen_users:
            user_net = network_list[user_number]
            user_net.load_state_dict(cn.state_dict())
            # print(f'Local user {user_number}: ', end='')
            for cn_param, user_gradient in zip(cn.named_parameters(),
                                               train(user_net, user_data[user_number],
                                                     user_target[user_number])):
                if cn_param[1].grad is not None:
                    cn_param[1].grad += user_gradient.clone() / active_user_count
                else:
                    cn_param[1].grad = user_gradient.clone() / active_user_count
        optimizer.step()
        optimizer.zero_grad()
    else:
        weights = []
        for user_number in chosen_users:
            user_net = network_list[user_number]
            user_net.load_state_dict(cn.state_dict())
            print(f'Local user {user_number}: ', end='')
            user_weight = train(user_net, user_data[user_number],
                                user_target[user_number])
            weights.append(user_weight)
        weights_average = {}
        first_user = True
        for user_weight in weights:
            for (key, value) in user_weight.items():
                if first_user:
                    weights_average[key] = value / active_user_count
                else:
                    weights_average[key] += value / active_user_count
            first_user = False
        cn.load_state_dict(weights_average)
    print(f'Global epoch {c_epoch}: ', end='')
    test_losses.append(test(cn))

fig = plt.figure()
plt.plot(test_losses)
plt.show()

# test_output_mat = test_output(cn)
# scipy.io.savemat('result1000.mat', {'test_output_mat': test_output_mat})
