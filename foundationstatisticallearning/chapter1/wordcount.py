import nltk
from nltk.corpus import webtext
from nltk.probability import FreqDist

nltk.download('punkt')

nltk.download('webtext')


text = webtext.raw('pirates.txt')
words = nltk.word_tokenize(text.lower())

stop_words = set(nltk.corpus.stopwords.words('english'))
filtered_words = [word for word in words if word.casefold() not in stop_words ]
freq_dist = nltk.FreqDist(filtered_words)
print(freq_dist.most_common(10))


# Remove punctuation
print("*********")
print("Remove punctuation")
filtered_words = [word for word in words if word.casefold() not in stop_words and word.isalpha()]
freq_dist_new = nltk.FreqDist(filtered_words)
print(freq_dist_new.most_common(10))

