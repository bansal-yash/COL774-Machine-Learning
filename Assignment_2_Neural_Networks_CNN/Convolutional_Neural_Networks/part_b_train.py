import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import trainloader
import argparse

torch.manual_seed(0)

# python3 part_b_train.py --train_dataset_root dataset/multi_dataset --save_weights_path part_b_multi_model.pth

parser = argparse.ArgumentParser()
parser.add_argument("--train_dataset_root", type=str)
parser.add_argument("--save_weights_path", type=str)
args = parser.parse_args()

root = args.train_dataset_root
weights_file = args.save_weights_path
train_csv = os.path.join(root, "public_train.csv")


class multi_cnn(nn.Module):
    def __init__(self):
        super(multi_cnn, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)

        self.fc1 = nn.Linear(128 * 11 * 24, 512)
        self.fc2 = nn.Linear(512, 8)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = x.view(-1, 128 * 11 * 24)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


train_dataset = trainloader.CustomImageDataset(root, train_csv, trainloader.transform)
train_dataloader = DataLoader(train_dataset, batch_size=128)

train_data_array = []
for idx, (images, labels) in enumerate(train_dataloader):
    train_data_array.append((images.float(), labels.float()))

multi_cnn_model = multi_cnn()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(multi_cnn_model.parameters(), lr=0.001)

num_epochs = 8

for curr_epoch in range(1, num_epochs + 1):
    multi_cnn_model.train()
    running_loss = 0.0

    for inputs, labels in train_data_array:
        optimizer.zero_grad()

        outputs = multi_cnn_model(inputs)
        outputs = outputs.squeeze()
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * (inputs.shape[0])

    running_loss /= 3200

torch.save(multi_cnn_model.state_dict(), weights_file)
