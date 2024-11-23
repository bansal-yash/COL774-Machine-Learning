import sys
import numpy as np
import os

computed_weights = sys.argv[1]
gold_weights = sys.argv[2]

def comment(s):
    print('Comment :=>> ' + s)


if os.path.exists(computed_weights) == False:
    comment("Weight file not created for part a")
    exit()

computed_weights = np.loadtxt(computed_weights)
gold_weights = np.loadtxt(gold_weights)

if(computed_weights.shape[0] != gold_weights.shape[0]):
    comment("Weight file of wrong dimensions for part a")

weight_val = 0

weight_error = np.sum(np.square(gold_weights - computed_weights))/np.sum(np.square(gold_weights))


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

comment("Part (a):")
comment("Weight normalized L2 Error for part (a): " + str(np.round(weight_error,decimals=5)))
comment("Grade for part (a) (tentative) = " + str(weight_val * 25) + " out of 25 for the given testcase")

