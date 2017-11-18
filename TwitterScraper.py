import requests
import json
import datetime
from abc import ABCMeta
from abc import abstractmethod
from urllib import parse
from bs4 import BeautifulSoup
from time import sleep
from concurrent.futures import ThreadPoolExecutor
import logging as log
import random
import re

__author__ = 'Daniele Bracciani'
# Thanks to Tom Dickinson


class TwitterSearch(metaclass=ABCMeta):
    
    outputFileName = "define_the_name_in_main"

    def __init__(self, rate_delay, error_delay=5):
        """
        :param rate_delay: How long to pause between calls to Twitter
        :param error_delay: How long to pause when an error occurs
        """
        self.rate_delay = rate_delay
        self.error_delay = error_delay

    def search(self, query):
        self.perform_search(query)

    def perform_search(self, query):
        """
        Scrape items from twitter
        :param query:   Query to search Twitter with. Takes form of queries constructed with using Twitters
                        advanced search: https://twitter.com/search-advanced
        """
        url = self.construct_url(query)
        continue_search = True
        min_tweet = None
        response = self.execute_search(url)
        while response is not None and continue_search and response['items_html'] is not None:
            tweets = self.parse_tweets(response['items_html'])

            # If we have no tweets, then we can break the loop early
            if len(tweets) == 0:
                break

            # If we haven't set our min tweet yet, set it now
            if min_tweet is None:
                min_tweet = tweets[0]

            continue_search = self.save_tweets(tweets)

            # Our max tweet is the last tweet in the list
            max_tweet = tweets[-1]
            if min_tweet['tweet_id'] is not max_tweet['tweet_id']:
                if "min_position" in response.keys():
                    max_position = response['min_position']
                else:
                    max_position = "TWEET-%s-%s" % (max_tweet['tweet_id'], min_tweet['tweet_id'])
                url = self.construct_url(query, max_position=max_position)
                # Sleep for our rate_delay
                sleep(self.rate_delay)
                response = self.execute_search(url)

    def execute_search(self, url):
        """
        Executes a search to Twitter for the given URL
        :param url: URL to search twitter with
        :return: A JSON object with data from Twitter
        """
        try:
            # Specify a user agent to prevent Twitter from returning a profile card
            headers = {
                'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.'
                              '86 Safari/537.36'
            }
            req = requests.get(url, headers=headers)
            # response = urllib2.urlopen(req)
            data = json.loads(req.text)
            return data

        # If we get a ValueError exception due to a request timing out, we sleep for our error delay, then make
        # another attempt
        except Exception as e:
            log.error(e)
            log.error("Sleeping for %i" % self.error_delay)
            sleep(self.error_delay)
            return self.execute_search(url)
    
    @staticmethod      
    def getHtmlFromUrl(url):
        try:
            # Specify a user agent to prevent Twitter from returning a profile card
            headers = {
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.'
                            '86 Safari/537.36'
            }
            req = requests.get(permalink, headers=headers)
            data = req.text
            #print("Sono in getHtmlFromUrl")
            return data

        except Exception as e:
            log.error("ERROR IN getHtmlFromUrl: " + e)
            print ("TROVATO ERRORE IN getHtmlFromUrl: " + str(e))

    @staticmethod 
    def getGeoFromHtml(html):
        try:
            soupGeo = BeautifulSoup(html, "html.parser")

            geo = soupGeo.find_all('a', class_="u-textUserColor js-nav js-geo-pivot-link")
            #print("Print Geo:" + str(geo))
            if geo is not None and len(geo) > 0:
                #print("Ritorno Valore Geo:" + str(geo[0].get_text()))
                return geo[0].get_text()
            else:
                return ""
        except Exception as e:
            print("ERRRROREEEEE: " + str(e))
            



    @staticmethod
    def parse_tweets(items_html):
        """
        Parses Tweets from the given HTML
        :param items_html: The HTML block with tweets
        :return: A JSON list of tweets
        """
        soup = BeautifulSoup(items_html, "html.parser")
        tweets = []
        #print("##################################")
        #print (str(items_html))
        #print("##################################")
        

        #########
        # emoticon;geo;mentions;hashtags;id;permalink
        ########
        
        for li in soup.find_all("li", class_='js-stream-item'):

            # If our li doesn't have a tweet-id, we skip it as it's not going to be a tweet.
            if 'data-item-id' not in li.attrs:
                continue

            tweet = {
                'tweet_id': li['data-item-id'],
                'text': None,
                'user_id': None,
                'user_screen_name': None,
                'user_name': None,
                'created_at': None,
                'retweets': 0,
                'favorites': 0,
                'emoticons' : None,
                'mentions': None,
                'hashtags': None,
                'location' : None,
                'permalink': None
            }


            '''
                emoticons = []
				for i in tweetPQ.items('p.js-tweet-text img.Emoji--forText'):
					emoticons.append(i.attr("alt"))
				
				retweets = int(tweetPQ("span.ProfileTweet-action--retweet span.ProfileTweet-actionCount").attr("data-tweet-stat-count").replace(",", ""));
				favorites = int(tweetPQ("span.ProfileTweet-action--favorite span.ProfileTweet-actionCount").attr("data-tweet-stat-count").replace(",", ""));
				dateSec = int(tweetPQ("small.time span.js-short-timestamp").attr("data-time"));
				id = tweetPQ.attr("data-tweet-id");
				permalink = tweetPQ.attr("data-permalink-path");
				
				# Geo part ----------------
				urlTweet = 'https://twitter.com' + permalink

            '''
            # emoticons patterns
            #try:
                # Wide UCS-4 build
            emoji_pattern = re.compile(u'['
                u'\U0001F300-\U0001F64F'
                u'\U0001F680-\U0001F6FF'
                u'\u2600-\u26FF\u2700-\u27BF]+', 
                re.UNICODE)
            '''except re.error:
                # Narrow UCS-2 build
                emoji_pattern = re.compile(u'('
                    u'\ud83c[\udf00-\udfff]|'
                    u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
                    u'[\u2600-\u26FF\u2700-\u27BF])+', 
                    re.UNICODE)'''
            # ---------------

            # Tweet Text, Mentions, Hashtags, Emoticons
            tweet['emoticons'] = []
            text_p = li.find("p", class_="tweet-text")
            if text_p is not None:
                txt = re.sub(r"\s+", " ", text_p.get_text().replace('# ', '#').replace('@ ', '@'))
                tweet['text'] = ascii(emoji_pattern.sub('', txt)) #Convert text in ascii and remove the emoji
                #tweet['text'] = text_p.get_text()
                tweet['mentions'] = " ".join(re.compile('(@\\w*)').findall(txt))
                tweet['hashtags'] = " ".join(re.compile('(#\\w*)').findall(txt))
                #------
                #print(txt)
                try:
                    listm = li.find("p",class_="TweetTextSize")
                    for elem in listm:
                        if elem is not None:
                            if len(elem) == 0:
                                tweet['emoticons'].append(ascii(elem['alt']))
 
                except Exception as e:
                    print ("TROVATO ERRORE IN emoticon: " + str(e))
                '''
                # Get the emoticons -------------
                emojis = emoji_pattern.findall(text_p.get_text())
                try:
                    for emoji in emojis:
                        s = str("" + emoji)
                        s = ascii(r''+s)
                        tweet['emoticons'].append(s)
                    if len(tweet['emoticons']) > 0:
                        print (str(emojis))
                except Exception as e:
                    print("Error getting emoticons: " + str(e))
                # --------------------------------
                '''

            # Tweet User ID, User Screen Name, User Name, Permalink
            user_details_div = li.find("div", class_="tweet")
            if user_details_div is not None:
                tweet['user_id'] = user_details_div['data-user-id']
                tweet['user_screen_name'] = user_details_div['data-screen-name']
                tweet['user_name'] = user_details_div['data-name']
                permalink = 'https://twitter.com' + user_details_div['data-permalink-path']
                tweet['permalink'] = permalink
                #---------------------------------------
                #htmlGeo = TwitterSearch.getHtmlFromUrl(permalink)
                # getHtmlFromUrl
                htmlGeo = ""
                try:
                    # Specify a user agent to prevent Twitter from returning a profile card
                    headers = {
                    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.'
                                    '86 Safari/537.36'
                    }
                    req = requests.get(permalink, headers=headers)
                    data = req.text
                    #print("Sono in getHtmlFromUrl")
                    htmlGeo = data

                except Exception as e:
                    log.error("ERROR IN getHtmlFromUrl: " + e)
                    print ("TROVATO ERRORE IN getHtmlFromUrl: " + str(e))
                # End getHtmlFromUrl ----

                # getGeoFromHtml
                try:
                    soupGeo = BeautifulSoup(htmlGeo, "html.parser")

                    geo = soupGeo.find_all('a', class_="u-textUserColor js-nav js-geo-pivot-link")
                    #print("Print Geo:" + str(geo))
                    if geo is not None and len(geo) > 0:
                        #print("Found a GEOOOOOOOO")
                        #print("Ritorno Valore Geo:" + str(geo[0].get_text()))
                        tweet['location'] = geo[0].get_text()
                    else:
                        tweet['location'] = ""
                except Exception as e:
                    print("ERRRROREEEEE: " + str(e))
                # End getGeoFromHtml
                #----------------------------------------
            # Tweet date
            date_span = li.find("span", class_="_timestamp")
            if date_span is not None:
                tweet['created_at'] = float(date_span['data-time-ms'])

            # Tweet Retweets
            retweet_span = li.select("span.ProfileTweet-action--retweet > span.ProfileTweet-actionCount")
            if retweet_span is not None and len(retweet_span) > 0:
                tweet['retweets'] = int(retweet_span[0]['data-tweet-stat-count'])

            # Tweet Favourites
            favorite_span = li.select("span.ProfileTweet-action--favorite > span.ProfileTweet-actionCount")
            if favorite_span is not None and len(retweet_span) > 0:
                tweet['favorites'] = int(favorite_span[0]['data-tweet-stat-count'])

            tweets.append(tweet)
        return tweets

    @staticmethod
    def construct_url(query, max_position=None):
        """
        For a given query, will construct a URL to search Twitter with
        :param query: The query term used to search twitter
        :param max_position: The max_position value to select the next pagination of tweets
        :return: A string URL
        """

        params = {
            # Type Param
            'f': 'tweets',
            # Query Param
            'q': query
        }

        # If our max_position param is not None, we add it to the parameters
        if max_position is not None:
            params['max_position'] = max_position

        url_tupple = ('https', 'twitter.com', '/i/search/timeline', '', parse.urlencode(params), '')
        return parse.urlunparse(url_tupple)

    @abstractmethod
    def save_tweets(self, tweets):
        """
        An abstract method that's called with a list of tweets.
        When implementing this class, you can do whatever you want with these tweets.
        """


class TwitterSearchImpl(TwitterSearch):

    def __init__(self, rate_delay, error_delay, max_tweets):
        """
        :param rate_delay: How long to pause between calls to Twitter
        :param error_delay: How long to pause when an error occurs
        :param max_tweets: Maximum number of tweets to collect for this example
        """
        super(TwitterSearchImpl, self).__init__(rate_delay, error_delay)
        self.max_tweets = max_tweets
        self.counter = 0
    
    def save_tweets(self, tweets):
        """
        Just prints out tweets
        :return:
        """
        
        for tweet in tweets:
            # Lets add a counter so we only collect a max number of tweets
            self.counter += 1

            if tweet['created_at'] is not None:
                t = datetime.datetime.fromtimestamp((tweet['created_at']/1000))
                fmt = "%Y-%m-%d %H:%M:%S"
                #log.info("%i [%s] - %s" % (self.counter, t.strftime(fmt), tweet['text']))

            # When we've reached our max limit, return False so collection stops
            if self.max_tweets is not None and self.counter >= self.max_tweets:
                return False

        return True
        

class TwitterSlicer(TwitterSearch):
    """
    Inspired by: https://github.com/simonlindgren/TwitterScraper/blob/master/TwitterSucker.py
    The concept is to have an implementation that actually splits the query into multiple days.
    The only additional parameters a user has to input, is a minimum date, and a maximum date.
    This method also supports parallel scraping.
    """
    def __init__(self, rate_delay, error_delay, since, until, n_threads=1):
        super(TwitterSlicer, self).__init__(rate_delay, error_delay)
        self.since = since
        self.until = until
        self.n_threads = n_threads
        self.counter = 0

    def search(self, query):
        n_days = (self.until - self.since).days
        tp = ThreadPoolExecutor(max_workers=self.n_threads)
        for i in range(0, n_days):
            since_query = self.since + datetime.timedelta(days=i)
            until_query = self.since + datetime.timedelta(days=(i + 1))
            day_query = "%s since:%s until:%s" % (query, since_query.strftime("%Y-%m-%d"),
                                                  until_query.strftime("%Y-%m-%d"))
            tp.submit(self.perform_search, day_query)
        tp.shutdown(wait=True)

    def save_tweets(self, tweets):
        """
        Just prints out tweets
        :return: True always
        """
        #outputFileName = "output_got" + str(random.random()) + ".txt"
        outputFile = open(TwitterSlicer.outputFileName,"a") 
        for tweet in tweets:
            # Lets add a counter so we only collect a max number of tweets
            self.counter += 1
            date = ""
            if tweet['created_at'] is not None:
                t = datetime.datetime.fromtimestamp((tweet['created_at']/1000))
                fmt = "%Y-%m-%d %H:%M:%S"
                #log.info("%i [%s] - %s" % (self.counter, t.strftime(fmt), tweet['text']))
                #print(str(tweet))
                date = t.strftime(fmt)
                outputFile.write(('\n%s;%s;%d;%d;"%s";%s;%s;%s;%s;%s;%s' % (tweet['user_screen_name'], t.strftime(fmt), tweet['retweets'], tweet['favorites'], tweet['text'],tweet['emoticons'],tweet['location'],tweet['mentions'],tweet['hashtags'],tweet['tweet_id'],tweet['permalink'])))
        outputFile.close()
        print("Last date analyzed: " + str(date))
        print("Number of tweets collected until now: " + str(self.counter))


        return True

    def initFile(fileName):
        outputFile = open(fileName,"w") 
        outputFile.write('username;date;retweets;favorites;text;emoticons;location;mentions;hashtags;id;permalink')
        outputFile.close()
        print ("File " + fileName + " initializated")

if __name__ == '__main__':
    log.basicConfig(level=log.INFO)

    #search_query = "#datamining2017daniele"
    search_query = '#cataluña OR #catalunya OR #independencia OR #independenciacatalunya OR #independenciadecataluña OR #viscacatalunya OR #independenciacataluña OR #puigdemont OR #barcelona OR #referéndum OR #carlespuigdemont OR #referendum OR #ciudadanos'  # this is what we're searching for
    rate_delay_seconds = 0
    error_delay_seconds = 5

    # Example of using TwitterSearch
    #twit = TwitterSearchImpl(rate_delay_seconds, error_delay_seconds, None)
    #twit.search(search_query)

    # Example of using TwitterSlice
    # Format date: yyyy-mm-dd
    sinceDate = "2017-08-30" # Lower bound date
    untilDate = "2017-09-30" # Upper bound date
    select_tweets_since = datetime.datetime.strptime(sinceDate, '%Y-%m-%d')
    select_tweets_until = datetime.datetime.strptime(untilDate, '%Y-%m-%d')
    threads = 10

    query = search_query
    query = query.replace("#", "_")
    if " " in query:
        query = query.split(" ")[0]
        
    TwitterSlicer.outputFileName = "output" + query + "_s_" + sinceDate + "_u_" + untilDate + ".csv"
    TwitterSlicer.initFile(TwitterSlicer.outputFileName)


    twitSlice = TwitterSlicer(rate_delay_seconds, error_delay_seconds, select_tweets_since, select_tweets_until,
                              threads)
    twitSlice.search(search_query)

    #print("TwitterSearch collected %i" % twit.counter)
    print("Tweet collected %i" % twitSlice.counter)