import pandas as pd
import numpy as np
import sys
import time
import math
from sklearn.preprocessing import StandardScaler

# python3 logistic.py a train1.csv params.txt modelweights.txt
# python3 logistic.py b train1.csv test1.csv modelweights.txt modelpredictions.csv


def makeU(XW):  # Tested
    u_helper_2 = np.exp(XW - np.max(XW, axis=1, keepdims=True))

    sums = np.zeros(u_helper_2.shape[0], dtype=float)

    for i in range(len(u_helper_2)):
        sums[i] = np.sum(u_helper_2[i])
        u_helper_2[i] = u_helper_2[i] / sums[i]

    return u_helper_2


def loss_f(x, y, w, frequency_array):  # Tested
    n = x.shape[0]
    xw = np.matmul(x, w)
    u = makeU(xw)

    logu = np.log(u)
    yt = y.T
    yt_logu = np.matmul(yt, logu) / frequency_array

    t = np.trace(yt_logu)
    loss_val = t / ((-2) * n)
    return loss_val


def constant_learning(x, y, alpha, epochs, batch_size, frequency_array):  # Tested
    n = x.shape[0]
    k = len(frequency_array)
    m = x.shape[1]
    w = np.zeros((m, k), dtype=np.float64)

    no_of_batches = math.ceil(n / batch_size)

    for i in range(epochs):
        print("epoch number :- ", i + 1)

        for j in range(no_of_batches):
            start = batch_size * j
            end = start + batch_size

            if j == no_of_batches - 1:
                end = n

            x_batch = x[start:end]
            y_batch = y[start:end].astype(float)
            x_batch_freq = np.zeros_like(x_batch)

            mask = y_batch == 1
            indices = np.argmax(mask, axis=1)
            x_batch_freq = x_batch / frequency_array[indices][:, np.newaxis]

            x_batch_w = x_batch @ w
            u = makeU(x_batch_w)

            u_minus_y = u - y_batch
            x_batch_freq_t = x_batch_freq.T
            xt_U_minus_Y = x_batch_freq_t @ u_minus_y

            gradient = xt_U_minus_Y / (2 * (end - start))
            w = w - (alpha * gradient)

    return w


def adaptive_learning(x, y, n0, klr, epochs, batch_size, frequency_array):  # Tested
    n = x.shape[0]
    k = len(frequency_array)
    m = x.shape[1]
    w = np.zeros((m, k), dtype=np.float64)

    no_of_batches = math.ceil(n / batch_size)

    for i in range(epochs):
        print("epoch number :- ", i + 1)
        alpha = n0 / (1 + klr * (i + 1))

        for j in range(no_of_batches):
            start = batch_size * j
            end = start + batch_size

            if j == no_of_batches - 1:
                end = n

            x_batch = x[start:end]
            y_batch = y[start:end].astype(float)
            x_batch_freq = np.zeros_like(x_batch)

            mask = y_batch == 1
            indices = np.argmax(mask, axis=1)
            x_batch_freq = x_batch / frequency_array[indices][:, np.newaxis]

            x_batch_w = x_batch @ w
            u = makeU(x_batch_w)

            u_minus_y = u - y_batch
            x_batch_freq_t = x_batch_freq.T
            xt_U_minus_Y = x_batch_freq_t @ u_minus_y

            gradient = xt_U_minus_Y / (2 * (end - start))
            w = w - (alpha * gradient)

    return w


def optimal_alpha(x, y, w, gradient, n0, frequency_array):  # Tested
    n_l = 0
    n_h = n0
    g = gradient
    l_start = loss_f(x, y, w, frequency_array)

    while l_start > (loss_f(x, y, (w - n_h * g), frequency_array)):
        n_h *= 2

    for _ in range(20):
        n1 = (2 * n_l + n_h) / 3
        n2 = (n_l + 2 * n_h) / 3
        l1 = loss_f(x, y, w - n1 * g, frequency_array)
        l2 = loss_f(x, y, w - n2 * g, frequency_array)

        if l1 > l2:
            n_l = n1
        elif l1 < l2:
            n_h = n2
        else:
            n_l = n1
            n_h = n2
    return (n_l + n_h) / 2


def ternary_search(x, y, n0, epochs, batch_size, frequency_array):  # Tested
    n = x.shape[0]
    k = len(frequency_array)
    m = x.shape[1]
    w = np.zeros((m, k), dtype=np.float64)

    no_of_batches = math.ceil(n / batch_size)

    for i in range(epochs):
        print("epoch number :- ", i + 1)

        for j in range(no_of_batches):

            start = batch_size * j
            end = start + batch_size

            if j == no_of_batches - 1:
                end = n

            x_batch = x[start:end]
            y_batch = y[start:end].astype(float)
            x_batch_freq = np.zeros_like(x_batch)

            mask = y_batch == 1
            indices = np.argmax(mask, axis=1)
            x_batch_freq = x_batch / frequency_array[indices][:, np.newaxis]

            x_batch_w = x_batch @ w
            u = makeU(x_batch_w)

            u_minus_y = u - y_batch
            x_batch_freq_t = x_batch_freq.T
            xt_U_minus_Y = x_batch_freq_t @ u_minus_y

            gradient = xt_U_minus_Y / (2 * (end - start))

            alpha = optimal_alpha(x_batch, y_batch, w, gradient, n0, frequency_array)

            w = w - (alpha * gradient)

    return w


def ternary_search_b(
    x,
    y,
    n0,
    epochs,
    batch_size,
    frequency_array,
    x_test,
    weights_file,
    predictions_file,
):  # Tested
    n = x.shape[0]
    k = len(frequency_array)
    m = x.shape[1]
    w = np.zeros((m, k), dtype=np.float64)

    no_of_batches = math.ceil(n / batch_size)

    for t1 in range(epochs):

        if t1 == 5 and sys.argv[1] == "b":
            print("Reporting the results after 5th epoch")
            x_test_w = np.matmul(x_test, w)
            prediction_probabilities = makeU(x_test_w)

            with open(weights_file, "w") as file:
                for i in range(len(w)):
                    for j in range(len(w[0])):
                        file.write(str(w[i][j]) + "\n")

            with open(predictions_file, "w") as file:
                for i in range(x_test.shape[0]):
                    for j in range(k):
                        file.write(str(prediction_probabilities[i][j]))
                        if j != k - 1:
                            file.write(",")
                    file.write("\n")

        print("epoch number :- ", t1 + 1)

        for j in range(no_of_batches):

            start = batch_size * j
            end = start + batch_size

            if j == no_of_batches - 1:
                end = n

            x_batch = x[start:end]
            y_batch = y[start:end].astype(float)
            x_batch_freq = np.zeros_like(x_batch)

            mask = y_batch == 1
            indices = np.argmax(mask, axis=1)
            x_batch_freq = x_batch / frequency_array[indices][:, np.newaxis]

            x_batch_w = x_batch @ w
            u = makeU(x_batch_w)

            u_minus_y = u - y_batch
            x_batch_freq_t = x_batch_freq.T
            xt_U_minus_Y = x_batch_freq_t @ u_minus_y

            gradient = xt_U_minus_Y / (2 * (end - start))

            alpha = optimal_alpha(x_batch, y_batch, w, gradient, n0, frequency_array)

            w = w - (alpha * gradient)

    return w


def fa():  # Tested
    training_file = sys.argv[2]
    parameter_file = sys.argv[3]

    modelweights_file = sys.argv[4]

    with open(parameter_file, "r") as file:
        params = file.readlines()

    learn_strat = int(params[0])
    epochs = int(params[2])
    batch_size = int(params[3])

    train_data = pd.read_csv(training_file)
    frequency_counts = train_data.iloc[:, -1].value_counts()

    k = len(frequency_counts)

    frequency_array = np.zeros(k, dtype=int)
    for value in range(1, k + 1):
        frequency_array[value - 1] = frequency_counts.get(value, 0)

    n, m = train_data.shape
    ones = np.ones(n)
    train_data.insert(0, "dummy", ones)

    y = train_data.iloc[:, -1]
    y = pd.get_dummies(y).to_numpy()
    x = train_data.iloc[:, :-1].to_numpy()

    if learn_strat == 1:
        alpha = float(params[1])
        w = constant_learning(x, y, alpha, epochs, batch_size, frequency_array)

    elif learn_strat == 2:
        n0, klr = params[1].split(",")
        n0 = float(n0)
        klr = float(klr)

        w = adaptive_learning(x, y, n0, klr, epochs, batch_size, frequency_array)

    elif learn_strat == 3:
        n0 = float(params[1])
        w = ternary_search(x, y, n0, epochs, batch_size, frequency_array)

    with open(modelweights_file, "w") as file:
        for i in range(len(w)):
            for j in range(len(w[0])):
                file.write(str(w[i][j]) + "\n")


def fb():  # Tested
    training_file = sys.argv[2]
    testing_file = sys.argv[3]
    weights_file = sys.argv[4]
    predictions_file = sys.argv[5]

    train_data = pd.read_csv(training_file)

    frequency_counts = train_data.iloc[:, -1].value_counts()
    k = len(frequency_counts)

    frequency_array = np.zeros(k, dtype=int)
    for value in range(1, k + 1):
        frequency_array[value - 1] = frequency_counts.get(value, 0)

    y_train = train_data.iloc[:, -1]
    y_train = pd.get_dummies(y_train).to_numpy()
    x_train = train_data.iloc[:, :-1].to_numpy()

    x_test = pd.read_csv(testing_file).to_numpy()

    scalar = StandardScaler().fit(x_train)
    x_train = scalar.transform(x_train)
    x_test = scalar.transform(x_test)

    x_train = np.insert(x_train, 0, np.ones(x_train.shape[0]), axis=1)
    x_test = np.insert(x_test, 0, np.ones(x_test.shape[0]), axis=1)
    n = x_train.shape[0]

    w = ternary_search_b(
        x_train,
        y_train,
        np.float64(1e-5),
        8,
        n,
        frequency_array,
        x_test,
        weights_file,
        predictions_file,
    )

    x_test_w = np.matmul(x_test, w)
    prediction_probabilities = makeU(x_test_w)

    with open(weights_file, "w") as file:
        for i in range(len(w)):
            for j in range(len(w[0])):
                file.write(str(w[i][j]) + "\n")

    with open(predictions_file, "w") as file:
        for i in range(x_test.shape[0]):
            for j in range(k):
                file.write(str(prediction_probabilities[i][j]))
                if j != k - 1:
                    file.write(",")
            file.write("\n")


if __name__ == "__main__":
    t1 = time.time()
    a_or_b = sys.argv[1]

    if a_or_b == "a":
        fa()
    elif a_or_b == "b":
        fb()

    t2 = time.time()
    print("Time taken is :- {:.2f} secs".format(t2 - t1))
