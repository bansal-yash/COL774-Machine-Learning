import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import sys

training_file = sys.argv[1]
created_file = sys.argv[2]
selected_file = sys.argv[3]

train_data = pd.read_csv(training_file)
n, _ = train_data.shape
ones = np.ones(n)
train_data.insert(0, "Dummy", ones)
train_data = train_data.sort_values(by="Total Costs")
k = 0.09

n_rem = int(k * n)
if k != 0:
    train_data = train_data.iloc[:-n_rem]

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

means = train_data.groupby("Facility Name")["Total Costs"].mean()
train_data["Facility Name"] = train_data["Facility Name"].map(means).fillna(0)

means = train_data.groupby("Patient Disposition")["Total Costs"].mean()
train_data["Patient Disposition"] = (
    train_data["Patient Disposition"].map(means).fillna(0)
)

means = train_data.groupby("CCSR Diagnosis Description")["Total Costs"].mean()
train_data["CCSR Diagnosis Description"] = (
    train_data["CCSR Diagnosis Description"].map(means).fillna(0)
)

means = train_data.groupby("CCSR Procedure Description")["Total Costs"].mean()
train_data["CCSR Procedure Description"] = (
    train_data["CCSR Procedure Description"].map(means).fillna(0)
)

means = train_data.groupby("APR DRG Description")["Total Costs"].mean()
train_data["APR DRG Description"] = (
    train_data["APR DRG Description"].map(means).fillna(0)
)

means = train_data.groupby("APR MDC Description")["Total Costs"].mean()
train_data["APR MDC Description"] = (
    train_data["APR MDC Description"].map(means).fillna(0)
)

train_data["Emergency Department Indicator"] = train_data[
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

train_data = train_data.astype(float)
y = y.astype(float)

headers = train_data.columns.tolist()
s = len(headers)

with open(created_file, "w") as file:
    for header in headers:
        file.write(header + "\n")

with open(selected_file, "w") as file:
    for i in range(s):
        file.write("1\n")
