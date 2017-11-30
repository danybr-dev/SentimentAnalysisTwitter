import re as regex
from TwitterFunc.data_initialization import TwitterData_Initialize

class TwitterCleanuper:
    def iterate(self):
        for cleanup_method in [self.remove_urls,
                               self.remove_usernames,
                               self.remove_numbers,
                               self.remove_na,
                               self.remove_special_chars]:
            yield cleanup_method

    @staticmethod
    def remove_by_regex(tweets, regexp):
        tweets.loc[:, "text"].replace(regexp, "", inplace=True)
        return tweets
    
    '''
    @staticmethod
    def replace_by_regex(tweets, regexp):
        tweets.loc[:, "text"].sub(regexp, r'\1\2', inplace=True)
        return tweets

    def replace_two_or_more(self, tweets):
        return TwitterCleanuper.replace_by_regex(tweets, regex.compile(r"(.)\1{2,}"))
    '''

    def remove_urls(self, tweets):
        return TwitterCleanuper.remove_by_regex(tweets, regex.compile(r"http\S+"))

    def remove_na(self, tweets):
        return tweets[tweets["text"] != "Not Available"]

    def remove_special_chars(self, tweets):  # it unrolls the hashtags to normal words
        for remove in map(lambda r: regex.compile(regex.escape(r)), [",", ":", "\"", "=", "&", ";", "%", "$",
                                                                     "@", "%", "^", "*", "(", ")", "{", "}",
                                                                     "[", "]", "|", "/", "\\", ">", "<", "-",
                                                                     "!","¡","?","¿", ".", "'",
                                                                     "--", "---", "#","…"]):
            tweets.loc[:, "text"].replace(remove, "", inplace=True)
        return tweets

    def remove_some_special_char(self, tweets): # it unrolls the hashtags to normal words
        for remove in map(lambda r: regex.compile(regex.escape(r)), [",", ":", "\"", "=", "&", ";", "%", "$",
                                                                     "@", "%", "^", "*", "(", ")", "{", "}",
                                                                     "[", "]", "|", "/", "\\", ">", "<", "-",
                                                                      ".", "'","--", "---","…"]):
            tweets.loc[:, "text"].replace(remove, "", inplace=True)

    def remove_usernames(self, tweets):
        return TwitterCleanuper.remove_by_regex(tweets, regex.compile(r"@[^\s]+[\s]?"))

    def remove_numbers(self, tweets):
        return TwitterCleanuper.remove_by_regex(tweets, regex.compile(r"[\s]?[0-9]+\.?[0-9]*"))

class TwitterCleanuper2:
    def iterate(self):
        for cleanup_method in [self.remove_urls,
                               self.remove_usernames,
                               self.remove_numbers,
                               self.remove_na,
                               self.remove_some_special_chars]:
            yield cleanup_method

    @staticmethod
    def remove_by_regex(tweets, regexp):
        tweets.loc[:, "text"].replace(regexp, "", inplace=True)
        return tweets

    '''
    @staticmethod
    def replace_by_regex(tweets, regexp):
        tweets.loc[:, "text"].sub(regexp, r"\1\1", inplace=True)
        return tweets

    def replace_two_or_more(self, tweets):
        return TwitterCleanuper.replace_by_regex(tweets, regex.compile(r"(.)\1{2,}"))
    '''

    def remove_urls(self, tweets):
        return TwitterCleanuper.remove_by_regex(tweets, regex.compile(r"http\S+"))

    def remove_na(self, tweets):
        return tweets[tweets["text"] != "Not Available"]

    def remove_some_special_chars(self, tweets): # it unrolls the hashtags to normal words
        for remove in map(lambda r: regex.compile(regex.escape(r)), [",", ":", "\"", "=", "&", ";", "%", "$",
                                                                     "@", "%", "^", "*", "(", ")", "{", "}",
                                                                     "[", "]", "|", "/", "\\", ">", "<", "-",
                                                                      ".", "'","--", "---","…"]):
            tweets.loc[:, "text"].replace(remove, "", inplace=True)

    def remove_usernames(self, tweets):
        return TwitterCleanuper.remove_by_regex(tweets, regex.compile(r"@[^\s]+[\s]?"))

    def remove_numbers(self, tweets):
        return TwitterCleanuper.remove_by_regex(tweets, regex.compile(r"[\s]?[0-9]+\.?[0-9]*"))

class TwitterData_Cleansing(TwitterData_Initialize):
    def __init__(self, previous):
        self.processed_data = previous.processed_data
        
    def cleanup(self, cleanuper):
        t = self.processed_data
        for cleanup_method in cleanuper.iterate():
            if not self.is_testing:
                t = cleanup_method(t)
            else:
                if cleanup_method.__name__ != "remove_na":
                    t = cleanup_method(t)

        self.processed_data = t