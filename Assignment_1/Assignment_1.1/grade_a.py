import sys
import numpy as np
import os

out_inp = sys.argv[1]
weight_inp = sys.argv[2]
out_model = sys.argv[3]
weight_model = sys.argv[4]

def comment(s):
    print('Comment :=>> ' + s)

if os.path.exists(out_inp) == False:
    comment("Prediction file not created for part a")
    exit()
if os.path.exists(weight_inp) == False:
    comment("Weight file not created for part a")
    exit()

pred = np.loadtxt(out_inp)
weight = np.loadtxt(weight_inp)
pred_model = np.loadtxt(out_model)
weight_model = np.loadtxt(weight_model)

if(pred.shape[0] != pred_model.shape[0]):
    comment("Prediction file of wrong dimensions for part a")
    exit()
if(weight.shape[0] != weight_model.shape[0]):
    comment("Weight file of wrong dimensions for part a")
    exit()

pred_val = 0
weight_val = 0

pred_error = np.sum(np.square(pred - pred_model))/np.sum(np.square(pred_model))
weight_error = np.sum(np.square(weight - weight_model))/np.sum(np.square(weight_model))

if pred_error < 1e-3:
    pred_val = 1
elif pred_error < 1e-2:
    pred_val = 0.75
elif pred_error < 1e-1:
    pred_val = 0.5
elif pred_error < 2.5e-1:
    pred_val = 0.25
else:
    pred_val = 0

if weight_error < 1e-3:
    weight_val = 1
elif weight_error < 1e-2:
    weight_val = 0.75
elif weight_error < 1e-1:
    weight_val = 0.5
elif weight_error < 2.5e-1:
    weight_val = 0.25
else:
    weight_val = 0

if(pred_val < 1):
    t = np.argmax(np.square(pred - pred_model))
    comment("Maximum predicition error (predicted,expected) on row:" + str(t+1) + " - (" + str(np.round(pred[t],decimals=2)) + " , " + str(np.round(pred_model[t],decimals=2)) + ")")
comment("Part (a):")
comment("Prediction normalized L2 error for part (a): " + str(np.round(pred_error,decimals=5)))
comment("Weight normalized L2 Error for part (a): " + str(np.round(weight_error,decimals=5)))
comment("Grade for part (a) (tentative) = " + str(pred_val * 6.25 + weight_val * 6.25))

