import streamlit as st


# Nlp packages
import spacy
from textblob import TextBlob
from gensim.summarization import summarize


# sumy packages
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Summary function summy
def sumy_summarizer(docs):
	parser = PlaintextParser.from_string(docs, Tokenizer('english'))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document, sentences_count = 4)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result


def text_analyzer(my_text):
	nlp = spacy.load("en_core_web_sm")
	docs = nlp(my_text)
	#tokens = [token.text for token in docs]
	lemma_tokens = [('"Token":{},\n "Lemma":{}'.format(token.text, token.lemma_)) for token in docs]
	return lemma_tokens

@st.cache
def entity_analyzer(my_text):
	nlp = spacy.load("en_core_web_sm")
	docs = nlp(my_text)
	entities = [(entity.text, entity.label_) for entity in docs.ents]
	tokens = [token.text for token in docs]
	alldata = [('"Token":{},\n "Entities":{}'.format(tokens, entities))]
	return alldata



def main():
	'''NLP app will be coded here'''
	st.title("NLP APP : A Simple yet Powerfull App.")
	st.subheader("Natural Language Processing")

	# tokenization
	if st.checkbox("Show tokens and lemma"):
		st.text("Tokenizing your text")
		message = st.text_area("Enter your text","please type here")
		if st.button("Analyze"):
			nlp_result = text_analyzer(message)
			st.json(nlp_result)


	# Named entity
	if st.checkbox("Show Named Entities"):
		st.text("Extracting Entities from your text")
		message = st.text_area("Enter your text","please type here")
		if st.button("Extract"):
			nlp_result = entity_analyzer(message)
			st.json(nlp_result)


	# Sentiment analysis
	if st.checkbox("Show Sentiment Analysis"):
		st.text("Sentiment Analysis")
		message = st.text_area("Enter your text","please type here")
		if st.button("Analyze",key = 1):
			blob = TextBlob(message)
			result_sentiment = blob.sentiment
			st.success(result_sentiment)
			

	# text summarizer
	if st.checkbox("Show Text Summarizer"):
		st.text("Summarizing your text")
		message = st.text_area("Enter your text","please type here")
		summary_option = st.selectbox("Select Summarizer",('gensim','sumy'))
		if st.button("Summarize",key = 1):
			if summary_option == 'gensim':
				st.text("Using gensim...")
				summary_result = summarize(message, ratio = 0.35)
			elif summary_option == 'sumy':
				summary_result = sumy_summarizer(message)

			else:
				st.warning("Using Default Summarizer")
				st.text("Using gensin")
				summary_result = summarize(message, ratio = 0.35)
			st.success(summary_result) 

	st.sidebar.subheader("About The NLP APP")
	st.sidebar.text("Its an Natural Language Processing app built using various NLP packages and streamlit")
	st.sidebar.text("Thank you to streamlit team")	
	st.sidebar.subheader("Thanks for using the app.....")	

if __name__ == '__main__':
	main()