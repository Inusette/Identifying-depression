# Thesis
Identifying depression based on texts

The aim of this project is to create a tool that could help identify 
people suffering from clinical depression, using texts they produce. It has been 
observed that there are certain cues in text production 
that are specific to depressed patients ([source](http://www.aaai.org/Papers/ICWSM/2008/ICWSM08-020.pdf))


### Motivation

It is claimed that around 50-70% of people who commit suicide suffer from
clinical depression ([source](https://books.google.de/books?id=nD6VAFvKGC0C&pg=PA453&redir_esc=y#v=onepage&q&f=false)).
Major depression affects adults, teens, and children, and frequently goes 
undiagnosed and untreated. 

Since there's still a lot of stigma in society regarding mental 
disorders, generally people tend to find it difficult to admit to their
friends and family that they might be depressed. Instead many turn to the Internet,
as it provides both anonymity and support from other people going through similar
problems. Given the respective reasons, it can be assumed that there is plenty of
available data that could be analyzed and used for machine learning purposes. 

### Idea

The idea is to gather data from various forums devoted to depression, and train 
a neural network that would then be able to identify texts produced by people 
affected by clinical depression. 

### Data

Some of the trustworthy data is already available: the data from English depression forums 
was collected by Pennebaker et al.(2008). However, that dataset only contains about 400 
blog posts, which might not be enough for machine learning purposes. Therefore, I'm still looking for other
sources - so far, I'm collecting posts from the subreddsubredditit (Reddit is a social news aggregation, 
web content rating, and discussion website) devoted to depression. Reddit data might not be as 
reliable, because often people there self-diagnose themselves, but most of the posts still
convey a strong negative emotion. 

I'm also looking for data on other health problems (it would be interesting to compare or
attempt to classify)

I will, of course, pre-process the data before using it. It might also be necessary to 
set a fixed size to the texts when using them as input data. 

### Method

I plan to train a Recurrent Neural Network (Or a Convolutional Neural Network) 
for this particular task. 

As a side project, I will attempt to use an unsupervised approach (a clustering method) 
to perform a similar task (classifying texts into 2 groups: healthy and depressed). 

### Evaluation

Cross-validation is normally used to estimate the accuracy of a model created by the 
learning method of choice.

### Literature

- [Unsupervised Emotion Detection from Text using Semantic and Syntactic Relations](http://www.cs.yorku.ca/~aan/research/paper/Emo_WI10.pdf)
- [Using Hashtags to Capture Fine Emotion Categories from Tweets](http://www.saifmohammad.com/WebDocs/hashtagMK-CI-2.pdf)
- [Predicting Depression via Social Media](http://course.duruofei.com/wp-content/uploads/2015/05/Choudhury_Predicting-Depression-via-Social-Media_ICWSM13.pdf)
- [The Psychology of Word Use in Depression Forums in English and in Spanish](http://www.aaai.org/Papers/ICWSM/2008/ICWSM08-020.pdf)
- [Language use of depressed and depression-vulnerable college students](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.224.4752&rep=rep1&type=pdf)