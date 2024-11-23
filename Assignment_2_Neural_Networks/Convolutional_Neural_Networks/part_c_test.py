import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import testloader
import argparse
import pickle

torch.manual_seed(0)

# python3 part_c_test.py --test_dataset_root dataset/multi_dataset --load_weights_path part_c_multi_model.pth --save_predictions_path predictions.pkl

parser = argparse.ArgumentParser()
parser.add_argument("--test_dataset_root", type=str)
parser.add_argument("--load_weights_path", type=str)
parser.add_argument("--save_predictions_path", type=str)
args = parser.parse_args()

root = args.test_dataset_root
weights_file = args.load_weights_path
predictions_file = args.save_predictions_path
test_csv = os.path.join(root, "public_test.csv")


class multi_cnn(nn.Module):
    def __init__(self):
        super(multi_cnn, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.bn6 = nn.BatchNorm2d(1024)

        self.fc1 = nn.Linear(1024 * 3 * 10, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 8)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        x = self.pool6(F.relu(self.bn6(self.conv6(x))))

        x = x.view(-1, 1024 * 3 * 10)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


test_dataset = testloader.CustomImageDataset(root, test_csv, testloader.transform)
test_dataloader = DataLoader(test_dataset, batch_size=10000)
for idx, (test_images) in enumerate(test_dataloader):
    pass

multi_cnn_model = multi_cnn()
multi_cnn_model.load_state_dict(torch.load(weights_file, weights_only=True))
multi_cnn_model.eval()

with torch.no_grad():
    outputs = multi_cnn_model(test_images)
    outputs = outputs.squeeze()
    _, predictions = torch.max(outputs, 1)
    all_predictions = predictions.cpu().numpy()

with open(predictions_file, "wb") as f:
    pickle.dump(predictions, f)
