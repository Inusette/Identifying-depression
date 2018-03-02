import os
import re
from nltk.tokenize import word_tokenize


def process(sentence):
    """
    Processes the given text.
    Turns into lower case, removes numbers, separates punctuation from words
    """
    # remove case
    sentence = sentence.lower()

    # remove digits
    sentence = re.sub('\d', '', sentence)

    # tokenize
    tokens = word_tokenize(sentence)

    return tokens


# iterate the file names in the directory
# for filename in os.listdir("reddit_data"):
post = open('./reddit_data/01Syd.txt', 'r')

# create a list for tokenized sentences
processed_post = []

# read the post line by line
for line in post:
    # check if the line isn't empty
    if line.strip():
        # tokenize the line
        words = process(line)
        # append the tokenized version to the list
        processed_post.append(words)

# create a new file with the same name
new_post = open('./reddit_data/01Syd.txt', "w")
string = ''

# write the tokenized post to the new file
for sent in processed_post:
    for tok in sent:
        #new_post.write(tok + " ")
        #print(tok, ' ')
        string += tok + ' '
print(string)

# close the file
new_post.close()
post.close()



