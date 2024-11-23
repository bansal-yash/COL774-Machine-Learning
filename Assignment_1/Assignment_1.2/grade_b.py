import sys
import numpy as np
import os
import pandas as pd
predicted_csv = sys.argv[1]
gold_csv = sys.argv[2]

def comment(s):
    print('Comment :=>> ' + s)


if os.path.exists(predicted_csv) == False:
    comment("Prediction csv not created for part (b)")
    exit()

y_pred = np.genfromtxt(predicted_csv, delimiter=',', dtype=None)
df_gold = pd.read_csv(gold_csv)
y_true = df_gold['Race'].to_numpy()


if (y_true.shape[0]!=y_pred.shape[0] or y_pred.shape[1]!=4):
    comment("Prediction file of wrong dimensions for part (b)")
    exit()


#Computing frequencies of each class in the test set
freq1 = np.count_nonzero(y_true==1)
freq2 = np.count_nonzero(y_true==2)
freq3 = np.count_nonzero(y_true==3)
freq4 = np.count_nonzero(y_true==4)
freq = [freq1,freq2,freq3,freq4]

loss = 0

for i in range(y_true.shape[0]):
    #Computing loss for a particular sample.
    eps = 1e-12 # Small epsilon value to prevent division by 0.
    true_label = int(y_true[i])
    probability = y_pred[i][true_label-1]
    l = np.log(probability+eps)/freq[true_label-1]
    loss = loss + l

loss = -loss/(2*y_true.shape[0])


comment("Loss obtained on the test set for part (b): " + str(loss))

