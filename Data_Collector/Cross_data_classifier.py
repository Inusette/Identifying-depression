import os
import numpy as np
import random
from bunch import Bunch

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as prf_evaluator


# the path to the data directories
DEP = "./reddit_depression"
NON_DEP = "./reddit_non_depression"
TEST_DEP = "./blogs_depression"
TEST_NON_DEP = "./blogs_non_depression"

# analyzer parameter ('word' - for word tokens, 'char_wb' - for character tokens)
ANALYZER = "word"

# Logistic Regression penalty
PENALTY = "l1"

# number of K-fold splits
KFOLD_SPLITS = 10

# the average parameter for the precision/recall evaluator
EVAL_AVERAGE = "binary"


def process_post(file_path):
    """
    Reads a file and extracts the raw text
    :param file_path: the path to the file
    :return: the processed string of raw text
    """

    # open and read the file
    post = open(file_path, 'r')

    text = ''

    # read the post line by line
    for line in post:

        if line.strip():  # check if the line isn't empty
            # decode the line and append to the text string
            line = line.decode('latin-1').encode('utf8').strip()
            text += line + ' '

    return text


def construct_data(dep_fnames, non_dep_fnames, dep_dir, non_dep_dir):
    """
    Constructs the data bunch that contains file names, file paths, raw file texts, and targets of files
    :param dep_fnames: list of file names in depression directory
    :param non_dep_fnames: list of file names in non-depression directory
    :return: the constructed data bunch
    """
    # instantiate the data bunch
    data = Bunch()

    # join the 2 arrays of file names
    file_names = np.concatenate((dep_fnames, non_dep_fnames))

    # shuffle the data and add to the data bunch
    random.shuffle(file_names)

    # assign the shuffled file names array to data
    data.filenames = file_names

    # instantiate the lists for data attributes
    data.filepath = []  # path to files
    data.data = []  # raw texts
    data.target = []  # target category index

    # iterate the file names
    for index in range(len(file_names)):
        fn = file_names[index]

        # if the file belongs to depression cat
        if file_names[index] in dep_fnames:

            # append the corresponding index to the target attribute
            data.target.append(0)

            # find and append the path of the file to path attribute
            data.filepath.append(os.path.join(dep_dir, fn))

        # repeat for the other category
        else:
            data.target.append(1)
            data.filepath.append(os.path.join(non_dep_dir, fn))

        # get the path of the current file
        f_path = data.filepath[index]

        # read the file and pre-process the text
        post_text = process_post(f_path)

        # append it to the data attribute
        data.data.append(post_text)

    return data


def train_lr_model(x_train, y_train):
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


def evaluate(model, x_test, y_test):
    """
    Calculates 2 types of accuracies, precision, recall, f-score and support for the given fold
    :param model: trained model
    :param x_test: test set
    :param y_test: test label set
    :return: bunch with evaluation scores as attributes
    """

    # create the evaluation bunch for the given fold
    ev = Bunch()

    # calculate the cross validation
    ev.cv = model.score(x_test, y_test)

    # calculate the roc score
    y_pred_lr = model.predict_proba(x_test)[:, 1]
    ev.roc = roc_auc_score(y_test, y_pred_lr)

    # get the precision, recall, f-score
    ev.precision, ev.recall, ev.fscore, ev.supp = \
        prf_evaluator(y_test, model.predict(x_test), average=EVAL_AVERAGE)

    return ev


def classify_train_test(x_train, y_train, x_test, y_test):

    # fit the data to the logistic regression model
    lr_model = train_lr_model(x_train, y_train)

    return evaluate(lr_model, x_test, y_test)


# -------------------------------------------------------------------------------
# Process the data
# -------------------------------------------------------------------------------

print "\nProcessing data"

# lists of file names in both directories
dep_fnames = np.array(os.listdir(DEP))
non_dep_fnames = np.array(os.listdir(NON_DEP))

print "number of depression files: ", len(dep_fnames)
print "number of non-depression files: ", len(non_dep_fnames)

# Construct the data
data = construct_data(dep_fnames, non_dep_fnames, DEP, NON_DEP)

print "number of texts in data ", len(data.data)
print "targets for the first 10 files: ", data.target[:10]
print "number of targets of files in data", len(data.target)
print "first 2 files: ", data.data[:2]

# -------------------------------------------------------------------------------
# do the same for the test data


test_dep_fnames = np.array(os.listdir(TEST_DEP))
test_non_dep_fnames = np.array(os.listdir(TEST_NON_DEP))

print "number of test depression files: ", len(test_dep_fnames)
print "number of test non-depression files: ", len(test_non_dep_fnames)

# Construct the data
test_data = construct_data(test_dep_fnames, test_non_dep_fnames, TEST_DEP, TEST_NON_DEP)

print "number of texts in test data ", len(test_data.data)
print "targets for the first 10 test files: ", test_data.target[:10]
print "number of targets of files in test data", len(test_data.target)
print "first two files", test_data.data[:2]


# -------------------------------------------------------------------------------
# Vectors and Model
# -------------------------------------------------------------------------------

print "\ncreating data vectors"

print("\ncount vectors: \n")

# create the target labels array
labels = np.array(data.target)

# create the target labels array for test data
test_labels = np.array(test_data.target)

# --------------------------------------------
# Frequency count vectors
# --------------------------------------------

# instantiate the vectorizer
vectorizer = CountVectorizer(ngram_range=(1, 2), analyzer=ANALYZER, encoding='utf8')

# Learn a vocabulary dictionary of all tokens in the raw documents
vectorizer.fit(data.data)

# Transform documents to document-term matrix.
X_counts = vectorizer.transform(data.data)

# Using the vectorizer trained on training data, transform the test data
test_X_counts = vectorizer.transform(test_data.data)

print "Count vectors shape: ", X_counts.shape


print "Test count vectors shape: ", test_X_counts.shape

# fit the data to the model and get the accuracy scores
count_eval = classify_train_test(X_counts, labels, test_X_counts, test_labels)

print "\ncross validation: ", count_eval.cv
print "roc: ", count_eval.roc
print "precision for the first 10: ", count_eval.precision
print "recall for the first 10: ", count_eval.recall
print "fscore for the first 10: ", count_eval.fscore


# ---------------------------------------------
# Tf-idf vectors
# ---------------------------------------------

print("\nTf-idf vectors: \n")

# instantiate the vectorizer
tf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer=ANALYZER, encoding='utf8')

# Learn a vocabulary dictionary of all tokens in the raw documents
tf_vectorizer.fit(data.data)

# Transform documents to document-term matrix.
X_tfidf = tf_vectorizer.transform(data.data)

# Using the vectorizer trained on training data, transform the test data
test_X_tfidf = tf_vectorizer.transform(test_data.data)

print "Tfidf vectors shape: ", X_tfidf.shape
print "Test Tfidf vectors shape: ", test_X_tfidf.shape

# fit the data to the model and get the accuracy scores
tfidf_eval = classify_train_test(X_tfidf, labels, test_X_tfidf, test_labels)

print "\ncross validation: ", tfidf_eval.cv
print "roc: ", tfidf_eval.roc
print "precision for the first 10: ", tfidf_eval.precision
print "recall for the first 10: ", tfidf_eval.recall
print "fscore for the first 10: ", tfidf_eval.fscore
