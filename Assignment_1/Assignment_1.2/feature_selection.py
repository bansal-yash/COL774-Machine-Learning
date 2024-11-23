import pandas as pd
import numpy as np
import sys

# python3 logistic_competitive.py train2.csv test2.csv output.txt


def feature_select():
    training_file = sys.argv[1]
    created_file = sys.argv[2]
    selected_file = sys.argv[3]

    training_data = pd.read_csv(training_file)

    y_train = training_data.iloc[:, -1]
    y_train = pd.get_dummies(y_train, columns="Gender", prefix="Gender")

    x_train = training_data.iloc[:, :-1]

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

    x_train["Emergency Department Indicator"] = x_train[
        "Emergency Department Indicator"
    ].replace({1: 0, 2: 1})

    for f in to_one_hot:
        x_train = pd.get_dummies(x_train, columns=[f], prefix=f)

    x_train = x_train.astype(float)
    y_train = y_train.astype(float)

    n, _ = x_train.shape
    ones = np.ones(n)
    x_train.insert(0, "dummy", ones)

    headers = x_train.columns.tolist()
    s = len(headers)

    with open(created_file, "w") as file:
        for header in headers:
            file.write(header + "\n")

    with open(selected_file, "w") as file:
        for i in range(s):
            file.write("1\n")


if __name__ == "__main__":

    feature_select()
