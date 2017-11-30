from TwitterFunc.data_processing import TwitterData_TokenStem
import nltk
from collections import Counter
import pandas as pd 

class TwitterData_Wordlist(TwitterData_TokenStem):
    def __init__(self, previous):
        self.processed_data = previous.processed_data
        
    whitelist = []
    wordlist = []
        
    def build_wordlist(self, min_occurrences=2, max_occurences=100000, stopwords=nltk.corpus.stopwords.words("spanish"),
                       stopwords2=nltk.corpus.stopwords.words("english"), whitelist=None):
        self.wordlist = []; 
        whitelist = self.whitelist if whitelist is None else whitelist
        import os
        if os.path.isfile("data/wordlist.csv"):
            word_df = pd.read_csv("data/wordlist.csv")
            word_df = word_df[word_df["occurrences"] > min_occurrences]
            self.wordlist = list(word_df.loc[:, "word"])
            return

        words = Counter()
        for idx in self.processed_data.index:
            words.update(self.processed_data.loc[idx, "text"])

        for idx, stop_word in enumerate(stopwords2):
            if stop_word not in whitelist:
                del words[stop_word] 

        for idx, stop_word in enumerate(stopwords):
            if stop_word not in whitelist:
                del words[stop_word]  
        
        for k in list(words.keys()):
            if len(k) < 2:
                del words[k]

        word_df = pd.DataFrame(data={"word": [k for k, v in words.most_common()],
                                     "occurrences": [v for k, v in words.most_common()]},
                               columns=["word", "occurrences"])

        word_df.to_csv("data/wordlist.csv", index_label="idx")
        self.wordlist = [k for k, v in words.most_common() if min_occurrences < v ]
    
    
class TwitterData_BagOfWords(TwitterData_Wordlist):
    def __init__(self, previous):
        self.processed_data = previous.processed_data
        self.wordlist = previous.wordlist
    
    def build_data_model(self):
        label_column = []
        if not self.is_testing:
            label_column = ["label"]

        columns = label_column + list(
            map(lambda w: str(w) + "_bow",self.wordlist))
        labels = []
        rows = []
        for idx in self.processed_data.index:
            current_row = []

            if not self.is_testing:
                # add label
                current_label = self.processed_data.loc[idx, "sentiment"]
                labels.append(current_label)
                current_row.append(current_label)

            # add bag-of-words
            tokens = set(self.processed_data.loc[idx, "text"])
            for _, word in enumerate(self.wordlist):
                current_row.append(1 if word in tokens else 0)

            rows.append(current_row)

        self.data_model = pd.DataFrame(rows, columns=columns)
        self.data_labels = pd.Series(labels)
        return self.data_model, self.data_labels
