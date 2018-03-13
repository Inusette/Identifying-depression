import os
import re
import numpy as np
import random
from bunch import Bunch

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


# test size proportions
TEST_SIZE = 0.4

# Logistic Regression penalty
PENALTY = "l1"

# the path to the data directories
DEP = "./reddit_data"
FAM = "./reddit_data_family"

# number of files to use for classification
NUM_FILES = 400

# number of K-fold splits
KFOLD_SPLITS = 10


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

    print "Count vectors shape: ", x_counts.shape

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

    print "Tf-idf vectors shape: ", x_tfidf.shape

    return x_tfidf


def fit_lr_model(x_train, y_train):
    """
    Creates a logistic regression model and fits the given data to it
    :param x_train: train set to fir to model
    :param y_train: train label set to fit to model
    :return: the model
    """

    # define the logistic regression classifier model
    lr_clf = LogisticRegression(penalty=PENALTY, class_weight='balanced')

    # fit the data to the model
    lr_clf.fit(x_train, y_train)

    return lr_clf


def get_class_accuracy(x, y):
    """
    Splits the data into K stratified folds and calculates the accuracy means of a logistic regression classifier
    :param x: vector data to split into train/test sets
    :param y: target labels of the data
    :return: mean cross validation accuracy and roc accuracy
    """

    # Create the stratified fold splits model
    skf = StratifiedKFold(n_splits=KFOLD_SPLITS, shuffle=True)

    # create the accuracy means
    val_cv_mean = []
    val_roc_mean = []

    # iterate all the indices the split() method returns
    for indx, (train_indices, test_indices) in enumerate(skf.split(x, y)):
        # print the running fold
        print "Training on fold " + str(indx + 1) + "/10..."

        # Generate batches from indices
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # fit the data to the logistic regression model
        lr_model = fit_lr_model(x_train, y_train)

        # calculate the cross validation
        cross_val = lr_model.score(x_test, y_test)

        # calculate the roc score
        y_pred_lr = lr_model.predict_proba(x_test)[:, 1]
        roc = roc_auc_score(y_test, y_pred_lr)

        # append both values to the mean validation lists
        val_cv_mean.append(cross_val)
        val_roc_mean.append(roc)

    # return both accuracy means
    return np.mean(val_cv_mean), np.mean(val_roc_mean)


# -------------------------------------------------------------------------------
# Process the data
# -------------------------------------------------------------------------------

print "\nProcessing data"

# instantiate the data bunch
data = Bunch()

# lists of file names in both directories
dep_fnames = np.array(os.listdir(DEP)[:800])
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

print "len filenames ", len(file_names)

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


print "data length ", len(data.data)
print "target 10", data.target[:10]
print "target length", len(data.target)
print data.data[:2]


# -------------------------------------------------------------------------------
# Vectors
# -------------------------------------------------------------------------------

print "\ncreating data vectors"

print("\ncount vectors: \n")

# create the target labels array
labels = np.array(data.target)

# --------------------------------------------
# Frequency count vectors
# --------------------------------------------

# transform raw data into count vectors
X_counts = make_count_vectors(data.data)

# fit the data to the model and get the accuracy scores
count_cross_val, count_roc_auc = get_class_accuracy(X_counts, labels)

print "\ncross validation: ", count_cross_val
print "roc: ", count_roc_auc


# ---------------------------------------------
# Tf-idf vectors
# ---------------------------------------------

print("\nTf-idf vectors: \n")

# transform raw data into tf-idf vectors
X_tfidf = make_tfidf_vectors(data.data)

# fit the data to the model and get the accuracy scores
tfidf_cross_val, tfidf_roc_auc = get_class_accuracy(X_tfidf, labels)

print "\ncross validation: ", tfidf_cross_val
print "roc: ", tfidf_roc_auc
