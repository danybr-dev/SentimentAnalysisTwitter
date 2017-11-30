import sys
from os import path
import numpy as np
from PIL import Image
import wikipedia
from wordcloud import WordCloud, STOPWORDS
from unicodedata import normalize
from collections import Counter
import csv
import nltk

# get path to script's directory
currdir = path.dirname(__file__)

def getStopWords():
	return nltk.corpus.stopwords.words("spanish") + nltk.corpus.stopwords.words("english")

def getWordsFromFile(fileName, verbose = True):
	csvfile = open(fileName, 'r')
	reader = csv.reader(csvfile, delimiter=';')
	data = []
	for line in reader:
		text = line[4].split(" ") # get the text from its colums
		stopwords = getStopWords()
		for word in text:
			if word not in stopwords and word != "" and word != "…":
				data.append(word.lower()) # insert each word in data
	freqs = Counter(data)
	if verbose:
		print("Frequencies are: \n" + str(freqs.most_common(500)) + '\n')
	words = ""
	for word,v in freqs.most_common(500):
		words = words + "\n" + word
	return words


def get_wiki(query):
	# get best matching title for given query
	title = wikipedia.search(query)[0]

	# get wikipedia page for selected title
	page = wikipedia.page(title)
	return page.content


def create_wordcloud(text):

	# create set of stopwords	
	stopwords= getStopWords()

	# names of the masks
	maskNames = ["twitter.png","cloud.png"]

	for maskName in maskNames:
		icon = Image.open(maskName)
		mask = Image.new("RGB", icon.size, (255,255,255))
		mask.paste(icon,icon)

		# create numpy araay for wordcloud mask image
		mask = np.array(mask)
		
		#mask = np.array(Image.open(path.join(currdir, "twitter.png")))

		# create wordcloud object
		wc = WordCloud(background_color="white",
						max_words=200, 
						mask=mask,
						stopwords=stopwords)
		
		# generate wordcloud
		wc.generate(text)

		# save wordcloud
		wc.to_file(path.join(currdir, "wc_" + maskName))

if __name__ == "__main__":
	# get query
	#query = sys.argv[1]
	
	# file name of input
	fileName = "test_lan_ca.txt"

	# get text for given query
	#text = get_wiki(query)
	
	# get words from a train file
	words = getWordsFromFile(fileName,False) # Verbose = False, if True it will display the most common words

	# generate wordcloud
	create_wordcloud(words)
