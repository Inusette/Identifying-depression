# Thesis
Identifying depression based on texts

The aim of this project is to create a tool that could help identify 
people suffering from clinical depression, using texts they produce, 
since it has been noted that there are certain cues in text production 
that are specific to depressed patients. 
[source](http://www.aaai.org/Papers/ICWSM/2008/ICWSM08-020.pdf)


### Motivation:

It is claimed that around 50-70% of people who commit suicide suffer from
clinical depression.  ([source](https://books.google.de/books?id=nD6VAFvKGC0C&pg=PA453&redir_esc=y#v=onepage&q&f=false))
Major depression affects adults, teens, and children, and frequently goes 
undiagnosed and untreated. 

Since there's still a lot of stigma in the society regarding mental 
disorders, generally people tend to find it difficult to admit to their
friends and family that they might be depressed. Instead many turn to Internet,
as it provides both anonymity and support from other people going through similar
problems. Given the respective reasons, it can be assumed that there is plenty of
available data (texts produced by depressed people) to be able to analyze 
it and use it for machine learning purposes. 

### Idea
The idea is to gather data from various forums devoted to depression, and train a 
a neural network that would then be able to identify texts produced by people 
affected by clinical depression. 

### Method
I plan to train an RRN for this particular task. 

As a side project, I will attempt to use an unsupervised approach (a clustering method) 
to perform a similar task (classifying texts into 2 groups: healthy and depressed). 

