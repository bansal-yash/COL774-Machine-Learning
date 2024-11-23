import pandas as pd
import numpy as np
import sys
import time
import math
from sklearn.preprocessing import StandardScaler

# python3 logistic_competitive.py train2.csv test2.csv output.txt


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

            if j in range(10):
                print(loss_f(x_batch, y_batch, w, frequency_array))

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

            if j in range(10):
                print(loss_f(x_batch, y_batch, w, frequency_array))

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

    for _ in range(15):
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

            if j in range(10):
                print(loss_f(x_batch, y_batch, w, frequency_array))

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


def log_comp():  # Tested
    training_file = sys.argv[1]
    testing_file = sys.argv[2]
    output_file = sys.argv[3]

    training_data = pd.read_csv(training_file)

    frequency_counts = training_data.iloc[:, -1].value_counts()
    k = len(frequency_counts)

    frequency_array = np.zeros(k, dtype=int)
    frequency_array[0] = frequency_counts[-1]
    frequency_array[1] = frequency_counts[1]

    y_train = training_data.iloc[:, -1]
    y_train = pd.get_dummies(y_train, columns="Gender", prefix="Gender")

    x_train = training_data.iloc[:, :-1]

    x_test = pd.read_csv(testing_file)

    to_drop = [
        "Hospital Service Area",
        "Hospital County",
        "Operating Certificate Number",
        "Permanent Facility Id",
        "Zip Code - 3 digits",
        "Payment Typology 1",
        "Payment Typology 2",
        "Payment Typology 3",
    ]

    x_train.drop(
        to_drop,
        axis=1,
        inplace=True,
    )

    x_test.drop(
        to_drop,
        axis=1,
        inplace=True,
    )

    to_target = [
        "APR DRG Code",
    ]

    to_one_hot = [
        "Facility Name",
        "Age Group",
        "Race",
        "Ethnicity",
        "Type of Admission",
        "Patient Disposition",
        "CCSR Diagnosis Code",
        "CCSR Procedure Code",
        "APR MDC Code",
        "APR Severity of Illness Description",
        "APR Risk of Mortality",
        "APR Medical Surgical Description",
    ]

    for f in to_target:
        combined_train = x_train.copy()
        combined_train["Gender_-1"] = y_train["Gender_-1"]
        combined_train["Gender_1"] = y_train["Gender_1"]

        probs_neg1 = combined_train.groupby(f)["Gender_-1"].mean()
        probs_pos1 = combined_train.groupby(f)["Gender_1"].mean()
        probs_mean = (probs_neg1 - probs_pos1) / 2

        x_train[f] = x_train[f].map(probs_mean).fillna(0)
        x_test[f] = x_test[f].map(probs_mean).fillna(0)

    x_train["Emergency Department Indicator"] = x_train[
        "Emergency Department Indicator"
    ].replace({1: 0, 2: 1})

    x_test["Emergency Department Indicator"] = x_test[
        "Emergency Department Indicator"
    ].replace({1: 0, 2: 1})

    for f in to_one_hot:
        x_train = pd.get_dummies(x_train, columns=[f], prefix=f)
        x_test = pd.get_dummies(x_test, columns=[f], prefix=f)

    missing_cols = set(x_train.columns) - set(x_test.columns)
    for col in missing_cols:
        x_test[col] = 0

    extra_cols = set(x_test.columns) - set(x_train.columns)
    x_test.drop(columns=extra_cols, inplace=True)

    x_test = x_test.reindex(columns=x_train.columns)

    x_train = x_train.astype(float)
    x_test = x_test.astype(float)
    y_train = y_train.astype(float)

    print(x_train.shape)

    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test.to_numpy()

    scalar = StandardScaler().fit(x_train)
    x_train = scalar.transform(x_train)
    x_test = scalar.transform(x_test)

    x_train = np.insert(x_train, 0, np.ones(x_train.shape[0]), axis=1)
    x_test = np.insert(x_test, 0, np.ones(x_test.shape[0]), axis=1)
    n = x_train.shape[0]

    w = ternary_search(x_train, y_train, np.float64(1000), 25, n, frequency_array)

    x_test_w = np.matmul(x_test, w)
    prediction_probabilities = makeU(x_test_w)

    with open(output_file, "w") as file:
        for i in range(prediction_probabilities.shape[0]):
            if prediction_probabilities[i][0] >= 0.5:
                file.write(str(-1) + "\n")
            else:
                file.write(str(1) + "\n")


if __name__ == "__main__":
    t1 = time.time()

    log_comp()

    t2 = time.time()
    print("Time taken is :- {:.2f} secs".format(t2 - t1))
