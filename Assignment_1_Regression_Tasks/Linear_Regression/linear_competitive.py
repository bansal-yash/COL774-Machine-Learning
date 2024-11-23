import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import sys

training_file = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

train_data = pd.read_csv(training_file)
n, _ = train_data.shape
ones = np.ones(n)
train_data.insert(0, "Dummy", ones)
train_data = train_data.sort_values(by="Total Costs")
k = 0.09

n_rem = int(k * n)
if k != 0:
    train_data = train_data.iloc[:-n_rem]

test_data = pd.read_csv(test_file)
n, _ = test_data.shape
ones = np.ones(n)
test_data.insert(0, "Dummy", ones)

train_data.drop(
    [
        "Hospital Service Area",
        "Hospital County",
        "Operating Certificate Number",
        "Permanent Facility Id",
        "Zip Code - 3 digits",
        "CCSR Diagnosis Code",
        "CCSR Procedure Code",
        "APR DRG Code",
        "APR MDC Code",
        "APR Severity of Illness Code",
    ],
    axis=1,
    inplace=True,
)

test_data.drop(
    [
        "Hospital Service Area",
        "Hospital County",
        "Operating Certificate Number",
        "Permanent Facility Id",
        "Zip Code - 3 digits",
        "CCSR Diagnosis Code",
        "CCSR Procedure Code",
        "APR DRG Code",
        "APR MDC Code",
        "APR Severity of Illness Code",
    ],
    axis=1,
    inplace=True,
)

means = train_data.groupby("Facility Name")["Total Costs"].mean()
train_data["Facility Name"] = train_data["Facility Name"].map(means).fillna(0)
test_data["Facility Name"] = test_data["Facility Name"].map(means).fillna(0)

means = train_data.groupby("Patient Disposition")["Total Costs"].mean()
train_data["Patient Disposition"] = (
    train_data["Patient Disposition"].map(means).fillna(0)
)
test_data["Patient Disposition"] = test_data["Patient Disposition"].map(means).fillna(0)

means = train_data.groupby("CCSR Diagnosis Description")["Total Costs"].mean()
train_data["CCSR Diagnosis Description"] = (
    train_data["CCSR Diagnosis Description"].map(means).fillna(0)
)
test_data["CCSR Diagnosis Description"] = (
    test_data["CCSR Diagnosis Description"].map(means).fillna(0)
)

means = train_data.groupby("CCSR Procedure Description")["Total Costs"].mean()
train_data["CCSR Procedure Description"] = (
    train_data["CCSR Procedure Description"].map(means).fillna(0)
)
test_data["CCSR Procedure Description"] = (
    test_data["CCSR Procedure Description"].map(means).fillna(0)
)

means = train_data.groupby("APR DRG Description")["Total Costs"].mean()
train_data["APR DRG Description"] = (
    train_data["APR DRG Description"].map(means).fillna(0)
)
test_data["APR DRG Description"] = test_data["APR DRG Description"].map(means).fillna(0)

means = train_data.groupby("APR MDC Description")["Total Costs"].mean()
train_data["APR MDC Description"] = (
    train_data["APR MDC Description"].map(means).fillna(0)
)
test_data["APR MDC Description"] = test_data["APR MDC Description"].map(means).fillna(0)


train_data["Emergency Department Indicator"] = train_data[
    "Emergency Department Indicator"
].replace({1: 0, 2: 1})
test_data["Emergency Department Indicator"] = test_data[
    "Emergency Department Indicator"
].replace({1: 0, 2: 1})

y = train_data["Total Costs"]
train_data.drop(["Total Costs"], axis=1, inplace=True)


train_data = pd.get_dummies(train_data, columns=["Age Group"], prefix="Age Group")
train_data = pd.get_dummies(train_data, columns=["Gender"], prefix="Gender")
train_data = pd.get_dummies(train_data, columns=["Race"], prefix="Race")
train_data = pd.get_dummies(train_data, columns=["Ethnicity"], prefix="Ethnicity")
train_data = pd.get_dummies(
    train_data, columns=["Type of Admission"], prefix="Type of Admission"
)
train_data = pd.get_dummies(
    train_data,
    columns=["APR Severity of Illness Description"],
    prefix="APR Severity of Illness Description",
)
train_data = pd.get_dummies(
    train_data, columns=["APR Risk of Mortality"], prefix="APR Risk of Mortality"
)
train_data = pd.get_dummies(
    train_data,
    columns=["APR Medical Surgical Description"],
    prefix="APR Medical Surgical Description",
)
train_data = pd.get_dummies(
    train_data, columns=["Payment Typology 1"], prefix="Payment Typology 1"
)
train_data = pd.get_dummies(
    train_data, columns=["Payment Typology 2"], prefix="Payment Typology 2"
)
train_data = pd.get_dummies(
    train_data, columns=["Payment Typology 3"], prefix="Payment Typology 3"
)

test_data = pd.get_dummies(test_data, columns=["Age Group"], prefix="Age Group")
test_data = pd.get_dummies(test_data, columns=["Gender"], prefix="Gender")
test_data = pd.get_dummies(test_data, columns=["Race"], prefix="Race")
test_data = pd.get_dummies(test_data, columns=["Ethnicity"], prefix="Ethnicity")
test_data = pd.get_dummies(
    test_data, columns=["Type of Admission"], prefix="Type of Admission"
)
test_data = pd.get_dummies(
    test_data,
    columns=["APR Severity of Illness Description"],
    prefix="APR Severity of Illness Description",
)
test_data = pd.get_dummies(
    test_data, columns=["APR Risk of Mortality"], prefix="APR Risk of Mortality"
)
test_data = pd.get_dummies(
    test_data,
    columns=["APR Medical Surgical Description"],
    prefix="APR Medical Surgical Description",
)
test_data = pd.get_dummies(
    test_data, columns=["Payment Typology 1"], prefix="Payment Typology 1"
)
test_data = pd.get_dummies(
    test_data, columns=["Payment Typology 2"], prefix="Payment Typology 2"
)
test_data = pd.get_dummies(
    test_data, columns=["Payment Typology 3"], prefix="Payment Typology 3"
)

ls = [
    "Facility Name",
    "Patient Disposition",
    "CCSR Diagnosis Description",
    "CCSR Procedure Description",
    "APR DRG Description",
    "APR MDC Description",
]

for i in ls:
    for j in ls:
        train_data[i + " into " + j] = train_data[i] * train_data[j]
        test_data[i + " into " + j] = test_data[i] * test_data[j]


missing_cols = set(train_data.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0

extra_cols = set(test_data.columns) - set(train_data.columns)
test_data.drop(columns=extra_cols, inplace=True)

test_data = test_data.reindex(columns=train_data.columns)

train_data = train_data.astype(float)
test_data = test_data.astype(float)
y = y.astype(float)

train_data = train_data.to_numpy()
y = y.to_numpy()
test_data = test_data.to_numpy()


model = LinearRegression()
model.fit(train_data, y)
prediction = model.predict(test_data)
test_size = prediction.shape[0]

with open(output_file, "w") as file:
    for i in range(test_size):
        file.write(str(prediction[i]) + "\n")
