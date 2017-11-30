from TwitterFunc.data_cleasing import TwitterData_Cleansing
import nltk

class TwitterData_TokenStem(TwitterData_Cleansing):
    def __init__(self, previous):
        self.processed_data = previous.processed_data

    def stem(self, stemmer=nltk.SnowballStemmer('spanish')):
        def stem_and_join(row):
            row["text"] = list(map(lambda str: stemmer.stem(str.lower()), row["text"]))
            return row

        self.processed_data = self.processed_data.apply(stem_and_join, axis=1)
    
    '''
    def lemma(self):
        def lemma_and_join(row):
            row["text"] = list(map(lambda str: patEs.lemma(str.lower()), row["text"]))
            return row

        self.processed_data = self.processed_data.apply(lemma_and_join, axis=1)
    '''

    def tokenize(self, tokenizer=nltk.word_tokenize):
        def tokenize_row(row):
            row["text"] = tokenizer(row["text"])
            row["tokenized_text"] = [] + row["text"]
            return row

        self.processed_data = self.processed_data.apply(tokenize_row, axis=1)
