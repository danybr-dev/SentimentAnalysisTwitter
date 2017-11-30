from __future__ import division
import numpy as np
import pandas as pd
import nltk
from TwitterFunc.data_additional_features import TwitterData_ExtraFeatures

class TwitterData(TwitterData_ExtraFeatures):

    def build_final_model(self, word2vec_provider, similarity_columns, stopwords=nltk.corpus.stopwords.words("english")):
        whitelist = self.whitelist
        stopwords = list(filter(lambda sw: sw not in whitelist, stopwords))
        extra_columns = [col for col in self.processed_data.columns if col.startswith("number_of")]
        #similarity_columns = ["bad_similarity", "good_similarity", "information_similarity"]
        label_column = []
        if not self.is_testing:
            label_column = ["label"]

        columns = label_column + ["original_id"] + extra_columns + similarity_columns + list(map(lambda w: str(w) + "_bow",self.wordlist)) + list(range(0,6))
        
        '''
        print('----------------------------------------------------------------------')
        print('colums'+str(columns)+'\n'+str(len(columns)))
        print('----------------------------------------------------------------------\n\n')
        '''
        
        labels = []
        rows = []
        for idx in self.processed_data.index:
            current_row = []

            if not self.is_testing:
                # add label
                current_label = self.processed_data.loc[idx, "sentiment"]
                labels.append(current_label)
                current_row.append(current_label)

            current_row.append(self.processed_data.loc[idx, "id"])

            for _, col in enumerate(extra_columns):
                current_row.append(self.processed_data.loc[idx, col])

            # average similarities with words
            tokens = self.processed_data.loc[idx, "tokenized_text"]
            for main_word in map(lambda w: w.split("_")[0], similarity_columns):
                current_similarities = [abs(sim) for sim in
                                        map(lambda word: word2vec_provider.get_similarity(main_word, word.lower()), tokens) if
                                        sim is not None]
                if len(current_similarities) <= 1:
                    current_row.append(0 if len(current_similarities) == 0 else current_similarities[0])
                    continue
                max_sim = max(current_similarities)
                min_sim = min(current_similarities)

                print(str(current_similarities))
                if(float(max_sim - min_sim)>0):
                    current_similarities = [float(sim-min_sim)/float(max_sim-min_sim) for sim in
                                        current_similarities]  # normalize to <0;1>
                else:
                    current_similarities = [float(sim) for sim in
                                        current_similarities]
                current_row.append(np.array(current_similarities).mean())


            def mean(numbers):
                return float(sum(numbers)) / max(len(numbers), 1)
            # add word2vec vector
            averaged_word2vec = list()
            tokens = self.processed_data.loc[idx, "tokenized_text"]
            current_word2vec = []
            for _, word in enumerate(tokens):
                vec = word2vec_provider.get_vector(word.lower())
                #print('vector'+str(vec)+'\n')
                if vec is not None:
                    #print(str(mean(vec)))
                    #current_word2vec.append(vec)
                    averaged_word2vec.append(mean(vec))

            current_row += averaged_word2vec

            # add bag-of-words
            tokens = set(self.processed_data.loc[idx, "text"])
            for _, word in enumerate(self.wordlist):
                current_row.append(1 if word in tokens else 0)

            rows.append(current_row)
        print('----------------------------------------------------------------------')
        print('data model:'+str(columns))
        print('----------------------------------------------------------------------\n\n')
        self.data_model = pd.DataFrame(rows, columns=columns)
        self.data_model = self.data_model.drop(labels=range(0,6),axis=1)
        self.data_labels = pd.Series(labels)
        
        
        print('----------------------------------------------------------------------')
        print('data model:'+str(self.data_model))
        print('----------------------------------------------------------------------\n\n')
        
        print('----------------------------------------------------------------------')
        print('data labels:'+str(self.data_labels))
        print('----------------------------------------------------------------------\n\n')
        
        return self.data_model, self.data_labels
