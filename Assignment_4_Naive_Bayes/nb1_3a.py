import pandas as pd
import numpy as np
import argparse
from nltk.stem import PorterStemmer
from nltk import bigrams

# python3 nb1_3a.py --train train.tsv --test valid.tsv --out out.txt --stop stopwords.txt

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

unigrams = set(word for tokens in tokenized_train_sentences for word in tokens)
bigrams_list = set(
    bigram for tokens in tokenized_train_sentences for bigram in bigrams(tokens)
)
vocabulary = {
    **{word: idx for idx, word in enumerate(unigrams)},
    **{f"{w1} {w2}": idx + len(unigrams) for idx, (w1, w2) in enumerate(bigrams_list)},
}
vocab_size = len(vocabulary)


def create_feature_vector(tokens):
    feature_vector = np.zeros(vocab_size)
    for token in tokens:
        if token in vocabulary:
            feature_vector[vocabulary[token]] = 1
    for w1, w2 in bigrams(tokens):
        bigram = f"{w1} {w2}"
        if bigram in vocabulary:
            feature_vector[vocabulary[bigram]] = 1
    return feature_vector


def get_num_new_words(tokens):
    num_new = 0
    for token in tokens:
        if token not in vocabulary:
            num_new += 1
    for w1, w2 in bigrams(tokens):
        bigram = f"{w1} {w2}"
        if bigram not in vocabulary:
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

likelihoods_occ = {}
likelihoods_not_occ = {}
for c in classes:
    x_c = x_train[y_train == c]
    likelihood = (np.sum(x_c, axis=0) + 1) / (len(x_c) + 2)
    likelihoods_occ[c] = likelihood
    likelihoods_not_occ[c] = 1 - likelihood


def predict(sample, idx=None):
    posteriors = []
    for c in classes:
        log_like_occ = np.sum(np.log(likelihoods_occ[c]) * sample)
        log_like_not_occ = np.sum(np.log(likelihoods_not_occ[c]) * (1 - sample))
        if idx != None:
            to_mult = 1 if (x_test_num_new_words[idx] > 0) else 0
            like_not_seen = to_mult * np.log(1 / (2 + np.sum(y_train == c)))
            posteriors.append(
                log_like_occ + log_like_not_occ + log_prior_prob[c] + like_not_seen
            )
        else:
            posteriors.append(log_like_occ + log_like_not_occ + log_prior_prob[c])

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
