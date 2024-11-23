import numpy as np
import pandas as pd
import sys
import csv
from sklearn.linear_model import LogisticRegression

# python3 comp_decision_tree.py test train_real.csv val_real.csv test_real.csv prediction_real_competitive.csv weights_real_competitive.csv

class Node:
    
    def __init__(self, depth, x_train, y_train, id):
        self.depth = depth
        self.x_train = x_train
        self.y_train = y_train
        self.id = id

        self.w = None
        self.threshold = None
        self.prediction = None
        self.right : Node = None
        self.left : Node = None
        self.isleaf = False
        self.label = None
        self.misclassified = None

    def calc_opt_w(self):
        model = LogisticRegression(fit_intercept=False)
        model.fit(self.x_train, self.y_train)

        weights = model.coef_.flatten()
        bias = model.intercept_

        self.w = weights

    def calc_opt_b(self):
        self.prediction = self.x_train @ self.w
        sorted_predictions = np.sort(self.prediction)
        min_gini = float("inf")

        for i in range(1, len(sorted_predictions)):
            threshold = (sorted_predictions[i-1] + sorted_predictions[i])/2

            left_split = self.prediction<=threshold
            right_split = self.prediction>threshold

            left_true = self.y_train[left_split]
            right_true = self.y_train[right_split]

            l_total = len(left_true)
            r_total = len(right_true)
            
            if l_total == 0 or r_total == 0:
                continue
            
            l_plus = np.sum(left_true)
            l_minus = l_total - l_plus

            gini_left = 1 - (l_plus/l_total)**2 - (l_minus/l_total)**2

            r_plus = np.sum(right_true)
            r_minus = r_total - r_plus
            gini_right = 1 - (r_plus/r_total)**2 - (r_minus/r_total)**2

            gini = (l_total*gini_left)+(r_total*gini_right)

            if gini <= min_gini:
                min_gini = gini
                self.threshold = threshold

    def split(self):
        left_split = self.prediction<=self.threshold
        right_split = self.prediction>self.threshold

        x_train_left = self.x_train[left_split]
        x_train_right = self.x_train[right_split]
        y_train_left = self.y_train[left_split]
        y_train_right = self.y_train[right_split]

        return x_train_left, x_train_right, y_train_left, y_train_right

class ObliqueDecisionTree:

    def __init__(self, max_depth = 8, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.weights_and_thresholds = []
        self.root = None

    def train_node(self, node:Node):
        if (node.depth == self.max_depth) or (len(node.y_train) < self.min_samples_split) or (len(node.y_train) == np.sum(node.y_train) or (np.sum(node.y_train) == 0)):
            node.isleaf = True

            num_pos = np.sum(node.y_train)
            num_neg = len(node.y_train) - num_pos

            if num_neg >= num_pos:
                node.label = 0
            else:
                node.label = 1

        else:
            node.calc_opt_w()
            node.calc_opt_b()

            if node.threshold == None:
                node.isleaf = True

                num_pos = np.sum(node.y_train)
                num_neg = len(node.y_train) - num_pos

                if num_neg >= num_pos:
                    node.label = 0
                else:
                    node.label = 1

            else:
                self.weights_and_thresholds.append([node.id] + list(node.w) + [node.threshold])
                x_train_left, x_train_right, y_train_left, y_train_right = node.split()

                node.left = Node(node.depth + 1, x_train_left, y_train_left, 2*node.id)
                node.right = Node(node.depth + 1, x_train_right, y_train_right, 2*node.id+1)

                num_pos = np.sum(node.y_train)
                num_neg = len(node.y_train) - num_pos

                if num_neg >= num_pos:
                    node.label = 0
                else:
                    node.label = 1

                self.train_node(node.left)
                self.train_node(node.right)

    def train(self, x_train, y_train):
        self.root = Node(0, x_train, y_train, 1)
        self.train_node(self.root)

    def save_weights(self, weights_file):
        with open(weights_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.weights_and_thresholds)

    def prune_node(self, node:Node , x_val, y_val):
        node.misclassified = np.sum(y_val != node.label)
        # total = len(node.y_train) + len(y_val)
        # pos = np.sum(node.y_train == 1) + np.sum(y_val == 1)
        # neg = total - pos

        # if neg >= pos:
        #     node.label = 0
        # else:
        #     node.label = 1

        if not node.isleaf:
            val_predict = x_val @ node.w

            left_split = val_predict <= node.threshold
            right_split = val_predict > node.threshold

            x_val_left = x_val[left_split]
            x_val_right = x_val[right_split]
            y_val_left = y_val[left_split]
            y_val_right = y_val[right_split]
            self.prune_node(node.left, x_val_left, y_val_left)
            self.prune_node(node.right, x_val_right, y_val_right)
            left_child_misclass = node.left.misclassified
            right_child_misclass = node.right.misclassified
            child_misclass = left_child_misclass + right_child_misclass

            if child_misclass >= node.misclassified:
                node.isleaf = True
                node.left = None
                node.right = None
                node.w = None
                self.weights_and_thresholds = [sublist for sublist in self.weights_and_thresholds if ((sublist[0] != node.id) and (sublist[0] != 2*node.id) and (sublist[0] != 2*node.id+1)) ]
                node.threshold = None
            else:
                node.misclassified = child_misclass

    def prune(self, x_val, y_val):
        self.prune_node(self.root, x_val, y_val)

    def find_leaf(self, node:Node , x_i):
        if node.isleaf:
            return node.label
        else:
            if x_i @ node.w <= node.threshold:
                return self.find_leaf(node.left, x_i)
            else:
                return self.find_leaf(node.right, x_i)

    def predict(self, x_test, pred_file):
        predictions = []

        for i in x_test:
            predictions.append([self.find_leaf(self.root, i)])
        
        with open(pred_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(predictions)

        predictions = [pred[0] for pred in predictions]
        return predictions


def run():
    train_file = sys.argv[2]
    val_file = sys.argv[3]
    test_file = sys.argv[4]
    prediction_file = sys.argv[5]
    weights_file = sys.argv[6]

    train_data = pd.read_csv(train_file)
    x_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values

    val_data = pd.read_csv(val_file)
    x_val = val_data.iloc[:, :-1].values
    y_val = val_data.iloc[:, -1].values

    test_data = pd.read_csv(test_file)

    max_depth = 9
    print("depth: ", max_depth)

    if "target" in test_data.columns:
        test_data.drop("target", axis = 1, inplace=True)

    x_test = test_data.values
    odt_tree = ObliqueDecisionTree(max_depth = max_depth)

    odt_tree.train(x_train, y_train)
    odt_tree.prune(x_val, y_val)
    odt_tree.save_weights(weights_file)
    predictions = np.array(odt_tree.predict(x_test, prediction_file))

run()