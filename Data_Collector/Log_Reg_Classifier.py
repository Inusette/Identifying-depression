import os
import re
from nltk.tokenize import word_tokenize
import numpy as np
import random
from bunch import Bunch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer


# test size proportions
TEST_SIZE = 0.4

# Logistic Regression penalty
PENALTY = "l1"

# the path to the data directories
DEP = "./reddit_data"
FAM = "./reddit_data_family"

# number of files to use for classification
NUM_FILES = 400


def process_post(file_path):

    # open and read the file
    post = open(file_path, 'r')

    # create a list for tokenized sentences
    processed_post = []

    # read the post line by line
    for line in post:

        if line.strip():  # check if the line isn't empty
            # tokenize the line
            words = process_sent(line)

            # append the tokenized version to the list
            processed_post.append(words)

    text = ''

    # iterate the post tokens list and build the string
    for sent in processed_post:
        for tok in sent:
            text += tok + ' '

    return text


def process_sent(sentence):
    """
    Processes the given text and turns into lower case, removes numbers, separates punctuation from words
    :param sentence: the raw sentence
    :return: the list of sentence tokens
    """
    sentence = sentence.decode("utf8")

    # remove case
    sentence = sentence.lower()

    # remove digits
    sentence = re.sub('\d', '', sentence)

    # tokenize
    tokens = word_tokenize(sentence)

    return tokens


def make_count_vectors(raw_data):
    """
    transforms text data into count vectors
    :param raw_data: data to transform
    :return: the count vectors
    """
    # instantiate the vectorizer
    vectorizer = CountVectorizer()

    # transform the data into vectors
    x_counts = vectorizer.fit_transform(raw_data)

    print("Count vectors shape: ", x_counts.shape)

    return x_counts


def make_tfidf_vectors(raw_data):
    """
    transforms text data into tf-idf vectors
    :param raw_data: data to transform
    :return: the tf-idf vectors
    """
    # instantiate the vectorizer
    vectorizer = TfidfVectorizer()

    # transform the data into vectors
    x_tfidf = vectorizer.fit_transform(raw_data)

    print("Tf-idf vectors shape: ", x_tfidf.shape)

    return x_tfidf


def lr_classify(x_train, x_test, y_train, y_test):

    # define the logistic regression classifier model
    lr_clf = LogisticRegression(penalty=PENALTY)

    # fit the data to the model
    lr_clf.fit(x_train, y_train)

    # calculate the cross validation
    cross_val = lr_clf.score(x_test, y_test)

    # calculate the roc score
    y_pred_lr = lr_clf.predict_proba(x_test)[:, 1]
    roc = roc_auc_score(y_test, y_pred_lr)

    return cross_val, roc


# -------------------------------------------------------------------------------
# Process the data
# -------------------------------------------------------------------------------

# instantiate the data bunch
data = Bunch()

# lists of file names in both directories
dep_fnames = np.array(os.listdir(DEP)[:NUM_FILES])
fam_fnames = np.array(os.listdir(FAM)[:NUM_FILES])

# join the 2 arrays of file names
file_names = np.concatenate((dep_fnames, fam_fnames))

# shuffle the data and add to the data bunch
random.shuffle(file_names)

# assign the shuffled file names array to data
data.filenames = file_names

# instantiate the lists for data attributes
data.filepath = []  # path to files
data.data = []  # raw texts
data.target = []  # target category index

print("10 first file names ", file_names[:10])
print("len filenames ", len(file_names))

# iterate the file names
for index in range(len(file_names)):
    fn = file_names[index]

    # if the file belongs to depression cat
    if file_names[index] in dep_fnames:

        # append the corresponding index to the target attribute
        data.target.append(0)

        # find and append the path of the file to path attribute
        data.filepath.append(os.path.join(DEP, fn))

    # repeat for the other category
    else:
        data.target.append(1)
        data.filepath.append(os.path.join(FAM, fn))

    # get the path of the current file
    f_path = data.filepath[index]

    # read the file and pre-process the text
    post_text = process_post(f_path)

    # append it to the data attribute
    data.data.append(post_text)


print("data length ", len(data.data))
print("target 10", data.target[:10])
print("target length", len(data.target))
print(data.data[:3])


# -------------------------------------------------------------------------------
# Count vectors
# -------------------------------------------------------------------------------

# transform raw data into count vectors
X_counts = make_count_vectors(data.data)

# split vectors into train and test sets
X_count_train, X_count_test, y_count_train, y_count_test = train_test_split(
    X_counts, np.array(data.target), test_size=TEST_SIZE, random_state=0)

print("Count vectors: train and test: ", X_count_train.shape, y_count_train.shape)


# -------------------------------------------------------------------------------
# Tf-idf vectors
# -------------------------------------------------------------------------------

# transform raw data into tf-idf vectors
X_tfidf = make_tfidf_vectors(data.data)

# split vectors into train and test sets
X_tfidf_train, X_tfidf_test, y_tfidf_train, y_tfidf_test = train_test_split(
    X_tfidf, np.array(data.target), test_size=TEST_SIZE, random_state=0)

print("tfidf vectors: train and test: ", X_tfidf_train.shape, y_tfidf_train.shape)


# -------------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------------

# fit the data to the model and get the accuracy scores

# for the count vectors:
count_cross_val, count_roc_auc = lr_classify(X_count_train, X_count_test, y_count_train, y_count_test)

print("\ncount vectors: \n")
print("cross validation: ", count_cross_val)
print("roc: ", count_roc_auc)


# for the tf-idf vectors
tfidf_cross_val, tfidf_roc_auc = lr_classify(X_tfidf_train, X_tfidf_test, y_tfidf_train, y_tfidf_test)

print("\nTf-idf vectors: \n")
print("cross validation: ", tfidf_cross_val)
print("roc: ", tfidf_roc_auc)
