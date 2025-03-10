import pandas as pd
import numpy as np
import argparse
from nltk.stem import PorterStemmer

# python3 nb1_2.py --train train.tsv --test valid.tsv --out out.txt --stop stopwords.txt

parser = argparse.ArgumentParser()
parser.add_argument("--train", required=True, type=str)
parser.add_argument("--test", required=True, type=str)
parser.add_argument("--out", required=True, type=str)
parser.add_argument("--stop", required=True, type=str)
args = parser.parse_args()

train_tsv = args.train
test_tsv = args.test
out_path = args.out
stop_path = args.stop

train_data = pd.read_csv(train_tsv, delimiter="\t", header=None, quoting=3)
test_data = pd.read_csv(test_tsv, delimiter="\t", header=None, quoting=3)
train_data = train_data[[2, 1]]
test_data = test_data[[2, 1]]
train_data.rename(columns={2: "news", 1: "class"}, inplace=True)
test_data.rename(columns={2: "news", 1: "class"}, inplace=True)

with open(stop_path, "r") as file:
    stopwords = file.read().splitlines()

stop_words = set(stopwords)
stemmer = PorterStemmer()

tokenized_train_sentences = []
for text in train_data["news"]:
    words = text.lower().split()
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words]
    tokenized_train_sentences.append(filtered_words)
train_data["tokenized_news"] = tokenized_train_sentences

tokenized_test_sentences = []
for text in test_data["news"]:
    words = text.lower().split()
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words]
    tokenized_test_sentences.append(filtered_words)
test_data["tokenized_news"] = tokenized_test_sentences

vocabulary = set(word for tokens in train_data["tokenized_news"] for word in tokens)
vocabulary = list(vocabulary)
vocab_size = len(vocabulary)
word_to_index = {word: idx for idx, word in enumerate(vocabulary)}


def create_feature_vector(tokens):
    feature_vector = np.zeros(vocab_size)
    for token in tokens:
        if token in word_to_index:
            feature_vector[word_to_index[token]] += 1
    return feature_vector


def get_num_new_words(tokens):
    num_new = 0
    for token in tokens:
        if token not in word_to_index:
            num_new += 1
    return num_new


x_train = np.array(
    [create_feature_vector(tokens) for tokens in train_data["tokenized_news"]]
)
y_train = train_data["class"].values

x_test = np.array(
    [create_feature_vector(tokens) for tokens in test_data["tokenized_news"]]
)
x_test_num_new_words = np.array(
    [get_num_new_words(tokens) for tokens in test_data["tokenized_news"]]
)
y_test = test_data["class"].values

classes = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
log_prior_prob = {}

for c in classes:
    log_prior_prob[c] = np.log(float(np.sum(y_train == c)) / len(y_train))

word_likelihoods = {}
total_words_laplace = {}
for c in classes:
    x_c = x_train[y_train == c]
    word_counts = np.sum(x_c, axis=0) + 1
    total_words = np.sum(x_c) + vocab_size
    total_words_laplace[c] = total_words
    word_likelihoods[c] = word_counts / total_words


def predict(sample, idx=None):
    posteriors = []
    for c in classes:
        log_likelihood = np.sum(sample * np.log(word_likelihoods[c]))
        posterior = log_likelihood + log_prior_prob[c]
        if idx != None:
            to_mult = x_test_num_new_words[idx]
            like_not_seen = to_mult * np.log(1 / total_words_laplace[c])
            posteriors.append(posterior + like_not_seen)
        else:
            posteriors.append(posterior)

    posteriors = np.array(posteriors)
    posteriors = np.exp(posteriors)
    return posteriors


train_pred_probs = np.array([predict(sample) for sample in x_train])
test_pred_probs = np.array(
    [predict(sample, idx) for (idx, sample) in enumerate(x_test)]
)

train_predictions = np.array(
    [classes[idx] for idx in np.argmax(train_pred_probs, axis=1)]
)
test_predictions = np.array(
    [classes[idx] for idx in np.argmax(test_pred_probs, axis=1)]
)

train_accuracy = (np.sum(y_train == train_predictions) / len(train_predictions)) * 100
test_accuracy = (np.sum(y_test == test_predictions) / len(test_predictions)) * 100

with open(out_path, "w") as file:
    for i in range(len(test_predictions)):
        file.write(test_predictions[i])
        if i != len(test_predictions) - 1:
            file.write("\n")
