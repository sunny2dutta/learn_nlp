
import contractions
import string


def clean_sentence(text):
	text  = contractions.fix(text)
	#extend all contractions
	PUNCT_TO_REMOVE = string.punctuation
	ans = text.translate(str.maketrans(”, ”, PUNCT_TO_REMOVE))
	#remove all punctuation
	return ans


def clean_words(word):
	ans = word.lower()



