import sys
import numpy as np
import os
import pandas as pd


def comment(s):
    print('Comment :=>> ' + s)

out_inp = sys.argv[1]
out_gold = sys.argv[2]

if os.path.exists(out_inp) == False:
    comment("Prediction file not created for part c")
    exit()

pred = np.loadtxt(out_inp)
df = pd.read_csv(out_gold)
y_true = df['Total Costs'].to_numpy()

if(pred.shape[0] != y_true.shape[0]):
    comment("Prediction file of wrong dimensions for part c")
    exit()

percentile_value = 90 #Percentage samples to consider
errors = (y_true-pred)**2 #Computing the error vector
errors = np.sort(errors) 
elements = int(percentile_value*len(errors)/100) #Taking the best 90% errors
errors = errors[:elements]
average_square_error = np.mean(errors) ##Average of square errors
rmse = (average_square_error)**0.5 #Root mean square
comment("Objective Function obtained on the test set = " + str(rmse))