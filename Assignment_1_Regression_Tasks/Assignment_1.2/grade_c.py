import sys
import numpy as np
import os
import pandas as pd
predicted_txt = sys.argv[1]
gold_csv = sys.argv[2]

def comment(s):
    print('Comment :=>> ' + s)


if os.path.exists(predicted_txt) == False:
    comment("Prediction csv not created for part (b)")
    exit()

y_pred = np.genfromtxt(predicted_txt, delimiter=',', dtype=None)
df_gold = pd.read_csv(gold_csv)
y_true = df_gold['Gender'].to_numpy()


if (y_true.shape[0]!=y_pred.shape[0] or y_pred.ndim!=1):
    comment("Prediction file of wrong dimensions for part (c)")
    exit()

total_samples = y_true.shape[0]
correct = 0

for i in range(y_true.shape[0]):
    true = int(y_true[i])
    pred = int(y_pred[i])
    if (pred==true):
        correct+=1
accuracy = correct/total_samples


comment("Accuracy obtained on the test set for part (c): " + str(accuracy))

