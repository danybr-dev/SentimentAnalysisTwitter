
# coding: utf-8

# In[45]:


import plotly
from plotly import graph_objs
from IPython.display import Image
from IPython.display import display, HTML
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier as XGBoostClassifier
from TwitterFunc import *
from collections import Counter
import nltk
import pandas as pd
from time import time
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
import numpy as np

def log(x):
    #can be used to write to log file
    print(x)

def test_classifier(X_train, y_train, X_test, y_test, classifier):
    if __name__=='__main__':
        log("")
        log("===============================================")
        classifier_name = str(type(classifier).__name__)
        log("Testing " + classifier_name)
        now = time()
        list_of_labels = sorted(list(set(y_train)))
        model = classifier.fit(X_train, y_train)
        log("Learing time {0}s".format(time() - now))
        now = time()
        predictions = model.predict(X_test)
        log("Predicting time {0}s".format(time() - now))

        precision = precision_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
        recall = recall_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
        log("=================== Results ===================")
        log("             Negative    Neutral    Positive")
        log("F1       " + str(f1))
        log("Precision" + str(precision))
        log("Recall   " + str(recall))
        log("Accuracy " + str(accuracy))
        log("===============================================")

        return precision, recall, accuracy, f1

def cv(classifier, X_train, y_train):
    if __name__=='__main__':
        log("===============================================")
        classifier_name = str(type(classifier).__name__)
        now = time()
        log("Crossvalidating " + classifier_name + "...")
        accuracy = [cross_val_score(classifier, X_train, y_train, cv=3, n_jobs=1)]
        log("Crosvalidation completed in {0}s".format(time() - now))
        log("Accuracy: " + str(accuracy[0]))
        log("Average accuracy: " + str(np.array(accuracy[0]).mean()))
        log("===============================================")
        return accuracy


# plotly configuration
plotly.offline.init_notebook_mode()


# In[46]:


import random
seed = 666
random.seed(seed)


# In[47]:


auto_open = False
do_BernoulliNB = True
do_RandomForest = True
do_RandomForest_full_model = True
do_XGBoost = True
plot_all = True


# In[48]:



data = TwitterData_Initialize()
data.initialize("data/train.txt", is_spain = True)

print('----------------------------------------------------------------------')
print('initial data is\n'+str(data.processed_data.head(5)))
print('----------------------------------------------------------------------\n\n')


# In[ ]:


'''
First thing that can be done as soon as the data is loaded is to see the 
data distribution. The training set had the following distribution
'''

if plot_all:
    df = data.processed_data
    positive = len(df[df["sentiment"] == "positive"])
    negative = len(df[df["sentiment"] == "negative"])
    neutral = len(df[df["sentiment"] == "neutral"])
    dist = [
    graph_objs.Bar(
        x=['positive','negative','neutral'],
        y= [positive,negative, neutral],
    )]

    plotly.offline.iplot({
    "data":dist, 
    "layout":graph_objs.Layout(title="Sentiment type distribution in training set")})



# In[7]:


'''
Cleansing

For the purpose of cleansing, the TwitterCleanup class was created. 
It consists methods allowing to execute all of the tasks show in the list above. 
Most of those is done using regular expressions.
The class exposes it's interface through iterate() method - it yields every cleanup 
method in proper order.
'''
data = TwitterData_Cleansing(data)
data.cleanup(TwitterCleanuper())

print('----------------------------------------------------------------------')
print('data after cleansing\n'+str(data.processed_data.head(5)))
print('----------------------------------------------------------------------')


# In[ ]:


'''
Tokenization & stemming

For the text processing, nltk library is used. First, 
the tweets are tokenized using nlkt.word_tokenize and then, 
stemming is done using PorterStemmer as the tweets are 100% in spanish.
The Spanish text also has some problems. Nouns that don’t need to be stemmed, 
such as ‘lugar’ (place) and ‘tiempo’ (time), have been reduced to unnecessary base forms.
In addition, verbs that come from common roots are stemmed inconsistently. 
For example, the verb ‘quiero’ (I want) is reduced to ‘quier’
, but you can see that another form of this verb, ‘queremos’ (we want) would be stemmed to ‘quer’ below.
'''
data = TwitterData_TokenStem(data)           
data.tokenize()
data.stem()

print('----------------------------------------------------------------------')
print('data after tokenization & stemming\n'+str(data.processed_data.head(5)))
print('----------------------------------------------------------------------\n\n')


# In[ ]:


'''
Building the wordlist
The wordlist (dictionary) is build by simple count of occurences of every unique word across all of the training dataset.
Before building the final wordlist for the model, let's take a look at the non-filtered version
'''


words = Counter()
for idx in data.processed_data.index:
    words.update(data.processed_data.loc[idx, "text"])

print('----------------------------------------------------------------------')
print('the most common words are:\n'+str(words.most_common(10)))
print('----------------------------------------------------------------------\n\n')


# In[87]:


'''
The most common words (as expected) are the typical spanish stopwords. 
We will filter them out, however, as purpose of this analysis is to determine sentiment, 
words like "not" and "n't" can influence it greatly. Having this in mind, this word will be whitelisted.
'''

stopwords=nltk.corpus.stopwords.words("spanish")
whitelist = ['espa']
for idx, stop_word in enumerate(stopwords):
    if stop_word not in whitelist:
        del words[stop_word]

print('----------------------------------------------------------------------')
print('the most common words after delition of spanish stopwords are:\n'+str(words.most_common(10)))
print('----------------------------------------------------------------------\n\n')


# In[68]:


stopwords=nltk.corpus.stopwords.words("english")
whitelist = []
for idx, stop_word in enumerate(stopwords):
    if stop_word not in whitelist:
        del words[stop_word]
        
print('----------------------------------------------------------------------')
print('the most common words after delition of spanish and english stopwords are:\n'+str(words.most_common(10)))
print('----------------------------------------------------------------------\n\n')


# In[69]:


data = TwitterData_Wordlist(data)
data.build_wordlist()

if plot_all :
    words = pd.read_csv("data/wordlist.csv")
    x_words = list(words.loc[0:7,"word"])
    x_words.reverse()
    y_occ = list(words.loc[0:7,"occurrences"])
    y_occ.reverse()

    dist = [
        graph_objs.Bar(
            x=y_occ,
            y=x_words,
            orientation="h"
    )]
    
    plotly.offline.iplot({"data":dist, "layout":graph_objs.Layout(title="Top words in built wordlist")})


# In[70]:


data = TwitterData_BagOfWords(data)
bow, labels = data.build_data_model()

print('----------------------------------------------------------------------')
print('The bow\n'+str(bow.head(10)))
print('----------------------------------------------------------------------\n\n')


# In[71]:


if plot_all :

    grouped = bow.groupby(["label"]).sum()
    words_to_visualize = []
    sentiments = ['positive','negative']
    #get the most 7 common words for every sentiment
    for sentiment in sentiments:
        words = grouped.loc[sentiment,:]
        words.sort_values(inplace=True,ascending=False)
        for w in words.index[:10]:
            if w not in words_to_visualize:
                words_to_visualize.append(w)
                
                
    #visualize it
    plot_data = []
    for sentiment in sentiments:
        plot_data.append(graph_objs.Bar(
                x = [w.split("_")[0] for w in words_to_visualize],
                y = [grouped.loc[sentiment,w] for w in words_to_visualize],
                name = sentiment
        ))
        
    plotly.offline.iplot({
            "data":plot_data,
            "layout":graph_objs.Layout(title="Most common words across sentiments")
        })


# In[72]:


'''
Experiment 1: BOW + Naive Bayes
It is nice to see what kind of results we might get from such simple model. 
The bag-of-words representation is binary, so Naive Bayes Classifier seems 
like a nice algorithm to start the experiments.
The experiment will be based on 7:3 train:test stratified split.
'''

if do_BernoulliNB :
    X_train, X_test, y_train, y_test = train_test_split(bow.iloc[:, 1:], bow.iloc[:, 0],
                                                        train_size=0.7, stratify=bow.iloc[:, 0],
                                                        random_state=seed)
    precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, BernoulliNB())
    nb_acc = cv(BernoulliNB(), bow.iloc[:,1:], bow.iloc[:,0])


# In[73]:


data = TwitterData_ExtraFeatures()
data.initialize("data/train.txt")
data.build_features()
data.cleanup(TwitterCleanuper())
data.tokenize()         
data.stem()
data.build_wordlist()
data_model, labels = data.build_data_model()

if plot_all:
    sentiments = ['positive','negative']
    plots_data_ef = []
    for what in map(lambda o: "number_of_"+o,["exclamation","hashtags","question"]):
        ef_grouped = data_model[data_model[what]>=1].groupby(["label"]).count()
        plots_data_ef.append({"data":[graph_objs.Bar(
                x = sentiments,
                y = [ef_grouped.loc[s,:][0] for s in sentiments],
        )], "title":"How feature \""+what+"\" separates the tweets"})
        
    i = 0
    for plot_data_ef in plots_data_ef:
        plotly.offline.iplot({
                "data":plot_data_ef["data"],
                "layout":graph_objs.Layout(title=plot_data_ef["title"])
        })



# In[74]:


if do_RandomForest :
    X_train, X_test, y_train, y_test = train_test_split(data_model.iloc[:, 1:], data_model.iloc[:, 0],
                                                        train_size=0.7, stratify=data_model.iloc[:, 0],
                                                        random_state=seed)
    precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, RandomForestClassifier(random_state=seed,n_estimators=403,n_jobs=-1))
    rf_acc = cv(RandomForestClassifier(n_estimators=403,n_jobs=-1, random_state=seed),data_model.iloc[:, 1:], data_model.iloc[:, 0])


# In[75]:


word2vec = Word2VecProvider()
# REPLACE PATH TO THE FILE
word2vec.load("data/embedding_file")


# In[77]:


similarity_columns = ["españa_similarity", "votarem_similarity", "catalexit_similarity"]
td = TwitterData()
td.initialize("data/train.txt")
td.build_features()
td.cleanup(TwitterCleanuper())
td.tokenize()
td.stem()
td.build_wordlist()
td.build_final_model(word2vec, similarity_columns= similarity_columns)

data_model = td.data_model
print(str(data_model.head(10)))


# In[ ]:


data_model.drop("original_id",axis=1,inplace=True)
columns_to_plot = similarity_columns
print(str(columns_to_plot))


# In[ ]:


if plot_all:
    #bad, good, info = columns_to_plot
    bad, good, info = columns_to_plot
    sentiments = ['positive','negative','neutral']


    only_positive = data_model[data_model[good]>=data_model[bad]]
    only_positive = only_positive[only_positive[good]>=only_positive[info]].groupby(["label"]).count()

    only_info = data_model[data_model[info]>=data_model[good]]
    only_info = only_info[only_info[info]>=only_info[bad]].groupby(["label"]).count()

    only_negative = data_model[data_model[bad] >= data_model[good]]
    only_negative = only_negative[only_negative[bad] >= only_negative[info]].groupby(["label"]).count()

    plot_data_w2v = []
    for sentiment in sentiments:
        plot_data_w2v.append(graph_objs.Bar(
                x = ["españa", "votarem", "catalexit"],
                y = [only_positive.loc[sentiment,:][0], only_negative.loc[sentiment,:][0], only_info.loc[sentiment,:][0]],
                name = "Number of dominating " + sentiment
        ))
        
    plotly.offline.iplot({
            "data":plot_data_w2v,
            "layout":graph_objs.Layout(title="Number of tweets dominating on similarity to: españa, votarem, catalexit")
        })


# In[ ]:


'''
Experiment 3: full model + Random Forest

The model is now complete. With a lot of new features, 
the learning algorithms should perform totally differently on the new data set.
'''
if do_RandomForest_full_model :
    X_train, X_test, y_train, y_test = train_test_split(data_model.iloc[:, 1:], data_model.iloc[:, 0],
                                                        train_size=0.7, stratify=data_model.iloc[:, 0],
                                                        random_state=seed)


    precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, RandomForestClassifier(n_estimators=403,n_jobs=-1, random_state=seed))
    rf_acc = cv(RandomForestClassifier(n_estimators=403,n_jobs=-1,random_state=seed),data_model.iloc[:, 1:], data_model.iloc[:, 0])


# In[ ]:


'''
Experiment 4: full model + XGBoost

XGBoost is relatively new machine learning algorithm based on decision trees and boosting. 
It is highly scalable and provides results, which are often higher than those obtained using popular algorithms 
such as Random Forest or SVM.
Important: XGBoost exposes scikit-learn interface, 
but it needs to be installed as an additional python package. 
See this page to see more: https://xgboost.readthedocs.io/en/latest/build.html
'''
if do_XGBoost :
    X_train, X_test, y_train, y_test = train_test_split(data_model.iloc[:, 1:], data_model.iloc[:, 0],
                                                        train_size=0.7, stratify=data_model.iloc[:, 0],
                                                        random_state=seed)
    precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, XGBoostClassifier(seed=seed))
    xgb_acc = cv(XGBoostClassifier(seed=seed),data_model.iloc[:, 1:], data_model.iloc[:, 0])

