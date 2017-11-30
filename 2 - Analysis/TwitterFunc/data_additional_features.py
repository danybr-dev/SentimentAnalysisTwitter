import pandas as pd
import re as regex
from TwitterFunc.data_bag_of_words import TwitterData_Wordlist

class TwitterData_ExtraFeatures(TwitterData_Wordlist):
    def __init__(self):
        pass
    
    def build_data_model(self):
        extra_columns = [col for col in self.processed_data.columns if col.startswith("number_of")]
        label_column = []
        if not self.is_testing:
            label_column = ["label"]

        columns = label_column + extra_columns + list(
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

            for _, col in enumerate(extra_columns):
                current_row.append(self.processed_data.loc[idx, col])

            # add bag-of-words
            tokens = set(self.processed_data.loc[idx, "text"])
            for _, word in enumerate(self.wordlist):
                current_row.append(1 if word in tokens else 0)

            rows.append(current_row)

        self.data_model = pd.DataFrame(rows, columns=columns)
        self.data_labels = pd.Series(labels)
        return self.data_model, self.data_labels
    
    def build_features(self):
        def count_by_lambda(expression, word_array):
            return len(list(filter(expression, word_array)))

        def count_occurences(character, word_array):
            counter = 0
            for j, word in enumerate(word_array):
                for char in word:
                    if char == character:
                        counter += 1

            return counter

        def count_by_regex(regex, plain_text):
            return len(regex.findall(plain_text))

        self.add_column("splitted_text", map(lambda txt: txt.split(" "), self.processed_data["text"]))

        # number of uppercase words
        uppercase = list(map(lambda txt: count_by_lambda(lambda word: word == word.upper(), txt),
                             self.processed_data["splitted_text"]))
        self.add_column("number_of_uppercase", uppercase)

        # number of !
        exclamations = list(map(lambda txt: count_occurences("!", txt)+count_occurences("¡", txt),
                                self.processed_data["splitted_text"]))
        self.add_column("number_of_exclamation", exclamations)

        # number of ?
        questions = list(map(lambda txt: count_occurences("?", txt)+count_occurences("¿", txt),
                             self.processed_data["splitted_text"]))
        self.add_column("number_of_question", questions)

        # number of ...
        ellipsis = list(map(lambda txt: count_by_regex(regex.compile(r"\.\s?\.\s?\."), txt),
                            self.processed_data["text"]))

        self.add_column("number_of_ellipsis", ellipsis)

        # number of hashtags
        hashtags = list(map(lambda txt: count_occurences("#", txt),
                            self.processed_data["splitted_text"]))

        self.add_column("number_of_hashtags", hashtags)

        # number of mentions
        mentions = list(map(lambda txt: count_occurences("@", txt),
                            self.processed_data["splitted_text"]))

        self.add_column("number_of_mentions", mentions)

        # number of quotes
        quotes = list(map(lambda plain_text: int(count_occurences("'", [plain_text.strip("'").strip('"')]) / 2 +
                                                 count_occurences('"', [plain_text.strip("'").strip('"')]) / 2),
                          self.processed_data["text"]))

        self.add_column("number_of_quotes", quotes)

        # number of urls
        urls = list(map(lambda txt: count_by_regex(regex.compile(r"http.?://[^\s]+[\s]?"), txt),
                        self.processed_data["text"]))

        self.add_column("number_of_urls", urls)

        '''
        # number of positive emoticons
        ed = EmoticonDetector()
        positive_emo = list(
            map(lambda txt: count_by_lambda(lambda word: ed.is_emoticon(word) and ed.is_positive(word), txt),
                self.processed_data["splitted_text"]))

        self.add_column("number_of_positive_emo", positive_emo)

        # number of negative emoticons
        negative_emo = list(map(
            lambda txt: count_by_lambda(lambda word: ed.is_emoticon(word) and not ed.is_positive(word), txt),
            self.processed_data["splitted_text"]))

        self.add_column("number_of_negative_emo", negative_emo)
        '''
    def add_column(self, column_name, column_content):
        self.processed_data.loc[:, column_name] = pd.Series(column_content, index=self.processed_data.index)

