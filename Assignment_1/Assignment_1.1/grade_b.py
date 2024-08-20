import sys
import numpy as np
import os


def comment(s):
    print('Comment :=>> ' + s)

out_inp = sys.argv[1]
weight_inp = sys.argv[2]
lambda_inp = sys.argv[3]
out_gold = sys.argv[4]
weight_gold = sys.argv[5]
lambda_gold = sys.argv[6]

if os.path.exists(out_inp) == False:
    comment("Prediction file not created for part b")
    exit()
if os.path.exists(weight_inp) == False:
    comment("Weight file not created for part b")
    exit()
if os.path.exists(lambda_inp) == False:
    comment("Best lambda file not created for part b")
    exit()

pred = np.loadtxt(out_inp)
weight = np.loadtxt(weight_inp)
pred_gold = np.loadtxt(out_gold)
weight_gold = np.loadtxt(weight_gold)
lambda_found = np.loadtxt(lambda_inp)
lambda_gold = np.loadtxt(lambda_gold)

if(pred.shape[0] != pred_gold.shape[0]):
    comment("Prediction file of wrong dimensions for part b")
    exit()
if(weight.shape[0] != weight_gold.shape[0]):
    comment("Weight file of wrong dimensions for part b")
    exit()

if (lambda_gold!=lambda_found):
    comment("Part (b): ")
    comment("Lambda found is incorrect")
    comment("True Lambda: " + str(lambda_gold))
    comment("Found Lambda: " + str(lambda_found))
    comment("Grade for part b) = 0.0")
    exit()

pred_val = 0
weight_val = 0

pred_error = np.sum(np.square(pred - pred_gold))/np.sum(np.square(pred_gold))
weight_error = np.sum(np.square(weight - weight_gold))/np.sum(np.square(weight_gold))

if pred_error < 1e-3:
    pred_val = 1
elif pred_error < 1e-2:
    pred_val = 0.8
elif pred_error < 1e-1:
    pred_val = 0.5
else:
    pred_val = 0

if weight_error < 1e-3:
    weight_val = 1
elif weight_error < 1e-2:
    weight_val = 0.8
elif weight_error < 1e-1:
    weight_val = 0.5
else:
    weight_val = 0
comment("Prediction normalized L2 error for part (b): " + str(np.round(pred_error,decimals=5)))
comment("Weight normalized L2 Error for part (b): " + str(np.round(weight_error,decimals=5)))
comment("Grade for part (b) (tentative) = " + str(pred_val * 6.25 + weight_val * 6.25))
    