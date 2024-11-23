import pandas as pd
import numpy as np
import argparse
from nltk.stem import PorterStemmer
from nltk import bigrams
import re

# python3 nb2.py --train train.tsv --test valid.tsv --out out.txt --stop stopwords.txt

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

train_data_raw = pd.read_csv(train_tsv, delimiter="\t", header=None, quoting=3)
test_data_raw = pd.read_csv(test_tsv, delimiter="\t", header=None, quoting=3)

train_data = train_data_raw.loc[:, [2, 1]]
test_data = test_data_raw.loc[:, [2, 1]]
train_data.columns = ["news", "class"]
test_data.columns = ["news", "class"]

with open(stop_path, "r") as file:
    stopwords = file.read().splitlines()

stop_words = set(stopwords)
porter_stemmer = PorterStemmer()

tokenized_train_sentences = []
for text in train_data["news"]:
    # words = text.lower().split()
    words = re.findall(r"\b\w+\b", text.lower())
    filtered_words = [
        porter_stemmer.stem(word) for word in words if word not in stop_words
    ]
    filtered_words = [word for word in filtered_words if word not in stop_words]
    tokenized_train_sentences.append(filtered_words)
train_data["tokenized_news"] = tokenized_train_sentences

tokenized_test_sentences = []
for text in test_data["news"]:
    # words = text.lower().split()
    words = re.findall(r"\b\w+\b", text.lower())
    filtered_words = [
        porter_stemmer.stem(word) for word in words if word not in stop_words
    ]
    filtered_words = [word for word in filtered_words if word not in stop_words]
    tokenized_test_sentences.append(filtered_words)
test_data["tokenized_news"] = tokenized_test_sentences


train_data["subjects"] = train_data_raw[3].apply(lambda x: x.split(","))
test_data["subjects"] = test_data_raw[3].apply(lambda x: x.split(","))

train_data["speaker"] = train_data_raw[4]
test_data["speaker"] = test_data_raw[4]

train_data["job"] = train_data_raw[5]
test_data["job"] = test_data_raw[5]

train_data["state"] = train_data_raw[6]
test_data["state"] = test_data_raw[6]

train_data["party"] = train_data_raw[7]
test_data["party"] = test_data_raw[7]

train_data["barely-true-counts"] = train_data_raw[8]
test_data["barely-true-counts"] = test_data_raw[8]

train_data["false-counts"] = train_data_raw[9]
test_data["false-counts"] = test_data_raw[9]

train_data["half-true-counts"] = train_data_raw[10]
test_data["half-true-counts"] = test_data_raw[10]

train_data["mostly-true-counts"] = train_data_raw[11]
test_data["mostly-true-counts"] = test_data_raw[11]

train_data["pants-fire-counts"] = train_data_raw[12]
test_data["pants-fire-counts"] = test_data_raw[12]


unigrams = set((word, 1) for tokens in tokenized_train_sentences for word in tokens)
bigrams_list = set(
    (f"{w1} {w2}", 1)
    for tokens in tokenized_train_sentences
    for (w1, w2) in bigrams(tokens)
)

subjects_list = set(
    ("sub" + subject, 8) for token in train_data["subjects"] for subject in token
)

speakers_list = set(("s" + speaker, 10) for speaker in train_data["speaker"])

speakers_hist_count = {
    "s" + speaker: [0, 0, 0, 0, 0] for speaker in train_data["speaker"]
}
speakers_curr_count = {
    "s" + speaker: [0, 0, 0, 0, 0, 0] for speaker in train_data["speaker"]
}


job_list = set(("j" + job, 1) for job in train_data["job"] if not pd.isna(job))
state_list = set(
    ("st" + state, 1) for state in train_data["state"] if not pd.isna(state)
)
party_list = set(
    ("p" + party, 1) for party in train_data["party"] if not pd.isna(party)
)


vocabulary = unigrams
vocabulary = vocabulary.union(bigrams_list)
vocabulary = vocabulary.union(subjects_list)
vocabulary = vocabulary.union(speakers_list)
vocabulary = vocabulary.union(job_list)
vocabulary = vocabulary.union(state_list)
vocabulary = vocabulary.union(party_list)

vocabulary = {word: [idx, f] for idx, (word, f) in enumerate(vocabulary)}
vocabulary = {word: [idx, val[1]] for idx, (word, val) in enumerate(vocabulary.items())}

vocab_size = len(vocabulary)


def create_feature_vector(row, train=False):
    tokenized_news = row["tokenized_news"]
    feature_vector = np.zeros(vocab_size)
    for token in tokenized_news:
        if token in vocabulary:
            feature_vector[vocabulary[token][0]] += vocabulary[token][1]

    for w1, w2 in bigrams(tokenized_news):
        bigram = f"{w1} {w2}"
        if bigram in vocabulary:
            feature_vector[vocabulary[bigram][0]] += vocabulary[bigram][1]

    for subject in row["subjects"]:
        subject = "sub" + subject
        if subject in vocabulary:
            feature_vector[vocabulary[subject][0]] += vocabulary[subject][1]

    speaker = "s" + row["speaker"]
    if speaker in vocabulary:
        feature_vector[vocabulary[speaker][0]] += vocabulary[speaker][1]

    speakers_hist_count[speaker] = [
        row["barely-true-counts"],
        row["false-counts"],
        row["half-true-counts"],
        row["mostly-true-counts"],
        row["pants-fire-counts"],
    ]

    if speaker not in speakers_curr_count:
        speakers_curr_count[speaker] = [0, 0, 0, 0, 0, 1]
    else:
        speakers_curr_count[speaker][5] += 1

    if train == True:
        if row["class"] == "pants-fire":
            speakers_curr_count[speaker][4] += 1
        if row["class"] == "false":
            speakers_curr_count[speaker][1] += 1
        if row["class"] == "mostly-true":
            speakers_curr_count[speaker][3] += 1
        if row["class"] == "barely-true":
            speakers_curr_count[speaker][0] += 1
        if row["class"] == "half-true":
            speakers_curr_count[speaker][2] += 1

    if not pd.isna(row["job"]):
        job = "j" + row["job"]
        if job in vocabulary:
            feature_vector[vocabulary[job][0]] += vocabulary[job][1]

    if not pd.isna(row["state"]):
        state = "st" + row["state"]
        if state in vocabulary:
            feature_vector[vocabulary[state][0]] += vocabulary[state][1]

    if not pd.isna(row["party"]):
        party = "p" + row["party"]
        if party in vocabulary:
            feature_vector[vocabulary[party][0]] += vocabulary[party][1]

    return feature_vector


x_train = np.array(
    [create_feature_vector(row, train=True) for (_, row) in train_data.iterrows()]
)
y_train = train_data["class"].values

x_test = np.array([create_feature_vector(row) for (_, row) in test_data.iterrows()])
y_test = test_data["class"].values

classes = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
log_prior_prob = {}

for c in classes:
    log_prior_prob[c] = np.log(float(np.sum(y_train == c)) / len(y_train))

alpha = 25
###############################################

word_likelihoods = {}
for c in classes:
    x_c = x_train[y_train == c]
    word_counts = np.sum(x_c, axis=0) + (1 * alpha)
    total_words = np.sum(x_c) + (vocab_size * alpha)
    word_likelihoods[c] = word_counts / total_words


def predict(sample, idx=None, test=False):
    posteriors = []
    for c in classes:
        log_likelihood = np.sum(sample * np.log(word_likelihoods[c]))
        posterior = log_likelihood + log_prior_prob[c]
        posteriors.append(posterior)

    posteriors = np.array(posteriors)

    if test == True:
        speaker = "s" + test_data["speaker"][idx]
        speakers_hist = speakers_hist_count[speaker]
        speakers_curr = speakers_curr_count[speaker]

        total_count = speakers_curr[5]
        # print(total_count)
        possible_false = speakers_hist[1] - speakers_curr[1]
        possible_mostly_true = speakers_hist[3] - speakers_curr[3]
        possible_barely_true = speakers_hist[0] - speakers_curr[0]
        possible_half_true = speakers_hist[2] - speakers_curr[2]
        possible_pants_fire = speakers_hist[4] - speakers_curr[4]

        possible_total = (
            total_count
            - speakers_curr[0]
            - speakers_curr[1]
            - speakers_curr[2]
            - speakers_curr[3]
            - speakers_curr[4]
        )

        posteriors = np.exp(posteriors)
        if possible_false == 0:
            posteriors[1] = 0
        if possible_mostly_true == 0:
            posteriors[4] = 0
        if possible_half_true == 0:
            posteriors[3] = 0
        if possible_pants_fire == 0:
            posteriors[0] = 0
        if possible_barely_true == 0:
            posteriors[2] = 0

    else:
        posteriors = np.exp(posteriors)
    return posteriors


train_pred_probs = np.array([predict(sample) for sample in x_train])
test_pred_probs = np.array(
    [predict(sample, idx, test=True) for (idx, sample) in enumerate(x_test)]
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
