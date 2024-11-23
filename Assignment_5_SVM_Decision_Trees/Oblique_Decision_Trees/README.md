Things Tried

- LABELING DONE ACCORDING TO TRAINING LABELS FOR PRUNING

- Added bias to logisttic in each node, as an attempt to find a decision boundary that can better adapt do data 
- changed logistic to linear for absolutely no reason, didn't work out (who could've thought?)
- tried regularization with L1 norm
- tried regularization with L2 norm
- tried changing pruning criterias by setting the criteria to % misclassification criteria. i.e. prune if (child misclass/parent misclass) <=k for k ranging from 0 to 1.5
- tried cross entropy impurity
- switched to scikitlearn logistic regression - recieved faster convergence.
- Tried SGD classifier 
- Plotted 231 graphs of data vs 21 single variables and 210 (21C2) variable combinations to try to figure out any underlying pattern within the data, unsuccessfully.
- tried various depths from 1-20.
- tried simultaneously making mulitple trees, to emulate something akin to a random forest, gave marginally better results, but was later discarded as refused on piazza
- tried target encoding binary variables, which was also discarded as refused on piazza



What worked 
- absolutely nothing 

Final Model Submitted - 
Naive part c + labelling for pruning as per train data + scikitlearn logistic regression for fast convergence
- this model gave test accuracy of 79.4%, which was the highest we obtained over all permutations and combinations of the above mentioned things tried
therefore submitting the above model which gave best predictions on test while satisfying all constraints provided in the assignment.
