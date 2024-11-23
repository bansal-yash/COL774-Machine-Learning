import numpy as np
import os
from preprocessor import CustomImageDataset, DataLoader, numpy_transform
import pickle
import argparse

np.random.seed(0)

# python3 part_b_multiclass.py --dataset_root dataset/multi_dataset --save_weights_path weights.pkl

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str)
parser.add_argument("--save_weights_path", type=str)
args = parser.parse_args()

dataset_root = args.dataset_root
weights_file = args.save_weights_path
csv = os.path.join(dataset_root, "train.csv")

batch_size = 256
learning_rate = 0.001
num_epochs = 15
num_layers = 4
layer_sizes = [625, 512, 256, 128, 8]

dataset = CustomImageDataset(root_dir=dataset_root, csv=csv, transform=numpy_transform)
dataloader = DataLoader(dataset, batch_size=batch_size)

weights_list = []
biases_list = []

for i in range(num_layers):
    weights_list.append(
        np.random.randn(layer_sizes[i], layer_sizes[i + 1])
        * np.sqrt(2 / layer_sizes[i])
    )
    biases_list.append(np.zeros(layer_sizes[i + 1]))

training_data_array = []

for x_batch, label_batch in dataloader:
    label_batch = np.eye(8)[label_batch.reshape(-1)]
    training_data_array.append((x_batch, label_batch))

for curr_epoch in range(1, num_epochs + 1):
    for x_batch, y_batch in training_data_array:
        curr_batch_size = x_batch.shape[0]

        # FORWARD PROPAGATION

        layer_activations = [x_batch]

        for i in range(1, num_layers):
            fc = (
                np.dot(layer_activations[i - 1], weights_list[i - 1])
                + biases_list[i - 1]
            )
            fc = 1 / (1 + np.exp(-fc))
            layer_activations.append(fc)

        out_layer = (
            np.dot(layer_activations[num_layers - 1], weights_list[num_layers - 1])
            + biases_list[num_layers - 1]
        )
        exp_out_layer = np.exp(out_layer - np.max(out_layer, axis=1, keepdims=True))
        out_layer = exp_out_layer / np.sum(exp_out_layer, axis=1, keepdims=True)

        # BACK PROPAGATION

        curr_da = None
        curr_dz = None

        for i in list(reversed(range(1, num_layers + 1))):
            if i == num_layers:
                curr_dz = (out_layer - y_batch) / curr_batch_size
            else:
                curr_dz = curr_da * layer_activations[i] * (1 - layer_activations[i])

            curr_dw = np.dot(np.transpose(layer_activations[i - 1]), curr_dz)
            curr_db = np.sum(curr_dz, axis=0)
            curr_da = np.dot(curr_dz, np.transpose(weights_list[i - 1]))

            weights_list[i - 1] -= learning_rate * curr_dw
            biases_list[i - 1] -= learning_rate * curr_db


final_dict = {
    "weights": {f"fc{i + 1}": weights_list[i] for i in range(num_layers)},
    "bias": {f"fc{i + 1}": biases_list[i] for i in range(num_layers)},
}

with open(weights_file, "wb") as file:
    pickle.dump(final_dict, file)
