import numpy as np
import os
from preprocessor import CustomImageDataset, DataLoader, numpy_transform
import pickle
import argparse

np.random.seed(0)

# python3 part_c_multi_best.py --dataset_root dataset/multi_dataset --save_weights_path weights.pkl

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str)
parser.add_argument("--save_weights_path", type=str)
args = parser.parse_args()

dataset_root = args.dataset_root
weights_file = args.save_weights_path
csv = os.path.join(dataset_root, "train.csv")

batch_size = 100
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
learning_rate = 0.001
num_epochs = 3000


num_layers = 5
layer_sizes = [625, 512, 256, 128, 32, 8]

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

m_w = [np.zeros_like(w) for w in weights_list]
v_w = [np.zeros_like(w) for w in weights_list]
m_b = [np.zeros_like(b) for b in biases_list]
v_b = [np.zeros_like(b) for b in biases_list]

curr_best_weight_list = weights_list.copy()
curr_best_biases_list = biases_list.copy()
curr_best_loss = np.inf

timestep = 0

for curr_epoch in range(1, num_epochs):
    if curr_epoch == 500:
        learning_rate = 0.0005
    if curr_epoch == 1000:
        learning_rate = 0.0003
    if curr_epoch == 1500:
        learning_rate = 0.0001
    if curr_epoch == 2000:
        learning_rate = 0.00005
        
    # FORWARD PROPAGATION
    temp_loss = 0
    for x_batch, y_batch in training_data_array:
        curr_batch_size = x_batch.shape[0]
        layer_activations = [x_batch]

        timestep += 1

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

        loss = -np.sum(y_batch * np.log(out_layer))
        temp_loss += loss

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

            m_w[i - 1] = beta1 * m_w[i - 1] + (1 - beta1) * curr_dw
            v_w[i - 1] = beta2 * v_w[i - 1] + (1 - beta2) * (curr_dw**2)
            m_b[i - 1] = beta1 * m_b[i - 1] + (1 - beta1) * curr_db
            v_b[i - 1] = beta2 * v_b[i - 1] + (1 - beta2) * (curr_db**2)

            m_w_hat = m_w[i - 1] / (1 - beta1**timestep)
            v_w_hat = v_w[i - 1] / (1 - beta2**timestep)
            m_b_hat = m_b[i - 1] / (1 - beta1**timestep)
            v_b_hat = v_b[i - 1] / (1 - beta2**timestep)

            weights_list[i - 1] -= (
                learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            )
            biases_list[i - 1] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    temp_loss /= 3200

    if temp_loss <= curr_best_loss:
        curr_best_weight_list = weights_list.copy()
        curr_best_biases_list = biases_list.copy()
        curr_best_loss = temp_loss

    if curr_epoch % 10 == 0:
        final_dict = {
            "weights": {
                f"fc{i + 1}": curr_best_weight_list[i] for i in range(num_layers)
            },
            "bias": {f"fc{i + 1}": curr_best_biases_list[i] for i in range(num_layers)},
        }

        with open(weights_file, "wb") as file:
            pickle.dump(final_dict, file)

# for x_batch, y_batch in training_data_array:
#     curr_batch_size = x_batch.shape[0]
#     layer_activations = [x_batch]

#     for i in range(1, num_layers):
#         fc = np.dot(layer_activations[i - 1], weights_list[i - 1]) + biases_list[i - 1]
#         fc = 1 / (1 + np.exp(-fc))
#         layer_activations.append(fc)

#     out_layer = (
#         np.dot(layer_activations[num_layers - 1], weights_list[num_layers - 1])
#         + biases_list[num_layers - 1]
#     )
#     exp_out_layer = np.exp(out_layer - np.max(out_layer, axis=1, keepdims=True))
#     out_layer = exp_out_layer / np.sum(exp_out_layer, axis=1, keepdims=True)

#     loss = -np.sum(y_batch * np.log(out_layer))
#     temp_loss += loss
# temp_loss /= 3200


final_dict = {
    "weights": {f"fc{i + 1}": curr_best_weight_list[i] for i in range(num_layers)},
    "bias": {f"fc{i + 1}": curr_best_biases_list[i] for i in range(num_layers)},
}

with open(weights_file, "wb") as file:
    pickle.dump(final_dict, file)
