import numpy as np
import pandas as pd
import sys

# python3 linear.py a train.csv test.csv sample_weights1.txt modelpredictions.txt modelweights.txt
# python3 linear.py b train.csv test.csv regularization.txt modelpredictions.txt modelweights.txt bestlambdafile.txt


a_or_b = sys.argv[1]
training_file = sys.argv[2]
test_file = sys.argv[3]


def fa():
    weights_file = sys.argv[4]
    predictions_file = sys.argv[5]
    model_weights_file = sys.argv[6]

    training_data = pd.read_csv(training_file)
    n, m = training_data.shape
    ones = np.ones(n)

    training_data.insert(0, "dummy", ones)

    # seperating x and y from the data
    y = training_data.iloc[:, -1].to_numpy()
    x = training_data.iloc[:, :-1].to_numpy()

    weights = pd.read_csv(weights_file, header=None).iloc[:, -1].to_numpy()

    # calculating weighted x and weighted y
    weighted_x = np.empty(x.shape)
    weighted_y = np.empty(y.shape)
    for i in range(n):
        weighted_x[i] = x[i] * weights[i]
        weighted_y[i] = y[i] * weights[i]

    transposed_weighted_x = np.transpose(weighted_x)  # this is xt_u

    xt = np.transpose(x)
    xt_u_x = np.matmul(transposed_weighted_x, x)

    xt_u_x_inv = np.linalg.inv(xt_u_x)

    xt_u_y = np.matmul(xt, weighted_y)

    # final model parameters
    w = np.matmul(xt_u_x_inv, xt_u_y)

    # now making predictions using the model
    test_data = pd.read_csv(test_file)
    test_size, _ = test_data.shape
    ones = np.ones(test_size)

    test_data.insert(0, "dummy", ones)

    # final prediction
    prediction = np.matmul(test_data, w)

    with open(model_weights_file, "w") as file:
        for i in range(m):
            file.write(str(w[i]) + "\n")

    with open(predictions_file, "w") as file:
        for i in range(test_size):
            file.write(str(prediction[i]) + "\n")


def fb():
    regular = sys.argv[4]
    mod_out = sys.argv[5]
    mod_wght = sys.argv[6]
    best_lamda_file = sys.argv[7]

    train_d = pd.read_csv(training_file)
    shap = train_d.shape
    n = shap[0]
    m = shap[1]

    ones = [1] * n
    train_d.insert(0, "b", ones)
    train_d.drop(index=train_d.index[(10 * (n // 10)) : n], inplace=True)
    n, _ = train_d.shape

    x = train_d.iloc[:, :-1]
    y = train_d.iloc[:, -1]

    with open(regular, "r") as file:
        lambdas = [float(line.strip()) for line in file]

    lamda_arr = []

    for lamda in lambdas:
        mse = 0
        for i in range(10):
            op = i * (n // 10)
            cl = (i + 1) * (n // 10)

            x_currtest = x.iloc[op:cl].to_numpy()
            x_currtrain = x.drop(index=x.index[op:cl]).to_numpy()

            y_currtest = y.iloc[op:cl].to_numpy()
            y_currtrain = y.drop(index=y.index[op:cl]).to_numpy()

            n_test, _ = x_currtest.shape

            xt = x_currtrain.T
            xt_x = np.matmul(xt, x_currtrain)
            for j in range(m):
                xt_x[j][j] += lamda

            xt_x_l_inv = np.linalg.inv(xt_x)
            xt_y = np.matmul(xt, y_currtrain)

            w = np.matmul(xt_x_l_inv, xt_y)

            prediction = x_currtest @ w
            error = y_currtest - prediction

            temp = np.dot(error, error)
            temp /= n_test
            mse += temp
        lamda_arr.append((mse, lamda))

    lamda_arr.sort()

    opt_lamda = lamda_arr[0][1]
    x = x.to_numpy()
    y = y.to_numpy()

    xt = x.T
    xt_x = np.matmul(xt, x)

    for j in range(m):
        xt_x[j, j] += opt_lamda

    xt_x_l_inv = np.linalg.inv(xt_x)

    xt_y = np.matmul(xt, y)

    w_final = np.matmul(xt_x_l_inv, xt_y)

    test_data = pd.read_csv(test_file)
    sze = test_data.shape[0]
    ones = [1] * sze

    test_data.insert(0, "b", ones)

    prediction = np.matmul(test_data, w_final)

    with open(mod_wght, "w") as file:
        for i in range(m):
            file.write(str(w_final[i]) + "\n")

    with open(mod_out, "w") as file:
        for i in range(sze):
            file.write(str(prediction[i]) + "\n")

    with open(best_lamda_file, "w") as file:
        file.write(str(opt_lamda))


if a_or_b == "a":
    fa()
elif a_or_b == "b":
    fb()
