import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import testloader
import argparse
import pickle

torch.manual_seed(0)

# python3 part_a_test.py --test_dataset_root dataset/binary_dataset --load_weights_path part_a_binary_model.pth --save_predictions_path predictions.pkl

parser = argparse.ArgumentParser()
parser.add_argument("--test_dataset_root", type=str)
parser.add_argument("--load_weights_path", type=str)
parser.add_argument("--save_predictions_path", type=str)
args = parser.parse_args()

root = args.test_dataset_root
weights_file = args.load_weights_path
predictions_file = args.save_predictions_path
test_csv = os.path.join(root, "public_test.csv")


class binary_cnn(nn.Module):
    def __init__(self):
        super(binary_cnn, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(64 * 12 * 25, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 25)
        x = self.fc1(x)
        return x


test_dataset = testloader.CustomImageDataset(root, test_csv, testloader.transform)
test_dataloader = DataLoader(test_dataset, batch_size=10000)
for idx, (test_images) in enumerate(test_dataloader):
    pass

binary_cnn_model = binary_cnn()
binary_cnn_model.load_state_dict(torch.load(weights_file, weights_only=True))
binary_cnn_model.eval()

with torch.no_grad():
    outputs = binary_cnn_model(test_images)
    outputs = outputs.squeeze()
    predictions = torch.round(torch.sigmoid(outputs))
    all_predictions = predictions.cpu().numpy()

with open(predictions_file, "wb") as f:
    pickle.dump(predictions, f)
