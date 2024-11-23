import numpy as np
import pandas as pd
import cvxpy
import json
import sys
from sklearn.preprocessing import StandardScaler

# python3 svm.py train_ls.csv

train_file = sys.argv[1]

train_data = pd.read_csv(train_file)
x_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
y_train = np.where(y_train == 0, -1, 1)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

num_sam = x_train.shape[0]
num_features = x_train.shape[1]

w = cvxpy.Variable(num_features)
b = cvxpy.Variable()
xi = cvxpy.Variable(num_sam)

objective = cvxpy.Minimize((0.5 * cvxpy.norm1(w)) + cvxpy.sum(xi))

constraints = [y_train[i] * (x_train[i] @ w + b) >= 1 - xi[i] for i in range(num_sam)]
constraints += [xi[i] >= 0 for i in range(num_sam)]

problem = cvxpy.Problem(objective, constraints)
problem.solve()

w, b, xi = w.value, b.value, xi.value
tol = 1e-4

margins = np.abs(y_train * (x_train @ w + b))
is_separable = np.all(xi <= tol)

if is_separable:
    support_vectors = np.where(np.abs(margins - 1) <= tol)[0].tolist()
else:
    support_vectors = []


base_name = train_file[:-4]
anystring = (base_name)[base_name.find("_") + 1 :]
weights_file = "weight_" + anystring + ".json"
weights_file_1 = "weights_" + anystring + ".json"
sv_file = "sv_" + anystring + ".json"

result = {"weights": w.tolist(), "bias": float(b)}
with open(weights_file, "w") as f:
    json.dump(result, f, indent=2)

with open(weights_file_1, "w") as f:
    json.dump(result, f, indent=2)

result = {"seperable": int(is_separable), "support_vectors": support_vectors}
with open(sv_file, "w") as f:
    json.dump(result, f, indent=2)
