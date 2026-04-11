from gensim.models import FastText
from gensim.models.word2vec import LineSentence
import numpy as np
import unicodedata
import joblib
import re

class Embedding_Vector ():

    def __init__ (self, token_vector_dimension):

        self.model = None
        self.token_vector_dimension = token_vector_dimension

    def get_vector_dimension (self):

        return self.token_vector_dimension

    def tokenize (self, text):
        
        text = text.lower()
        text = unicodedata.normalize ('NFKD', text)

        pattern = r"\w+(?:['-_]\w+)*|[^\w\s]+"
        text = re.findall (pattern, text)
        
        return text
    
    def build_vocabulary (self, file):

        with open ('processed_reviews.txt', 'w') as writer:
            for review in file:
                tokenized_review = self.tokenize (review)
                merged_tokenized_review = ' '.join (tokenized_review)
                writer.write (merged_tokenized_review + '\n')
        
        processed_file = LineSentence ('processed_reviews.txt') 
        self.model = FastText (sentences = processed_file, vector_size = self.token_vector_dimension, workers = 10)
            
    def vectorize (self, text, model_input_length):

        tokenized_text = self.tokenize (text)
        text_embedding_vector = []

        while len (tokenized_text) > model_input_length:
            del tokenized_text[-1]

        for token in tokenized_text:
            try:
                token_embedding_vector = self.model.wv[token]
                text_embedding_vector.append (token_embedding_vector)
            except KeyError:
                text_embedding_vector.append (np.zeros (self.token_vector_dimension))

        while len (text_embedding_vector) < model_input_length:
            text_embedding_vector.append (np.zeros (self.token_vector_dimension))

        return text_embedding_vector

    def X_setup (self, review_text, input_length):

        X = []
        for text in review_text:
            text_embedding_vector = self.vectorize (text, input_length)
            X.append (text_embedding_vector)
        
        return np.array (X)


if __name__ == '__main__':

    dataset = joblib.load ('setup_dataset.pkl')
    token_vector_dimension = 130
    token_vector_model = Embedding_Vector (token_vector_dimension)
    token_vector_model.build_vocabulary (dataset['review'])

    joblib.dump (token_vector_model, 'tokenization_vectorization_model.pkl')






# cd '/Users/chikhoado/Desktop/PROJECTS/Sentiment Analyzer'
# /opt/homebrew/bin/python3.12 -m venv .venv
# source .venv/bin/activate
# pip install --upgrade pip
# brew install openjdk@17 apache-spark
# sudo ln -sfn /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-17.jdk
# pip install numpy gensim regex joblib
# python '/Users/chikhoado/Desktop/PROJECTS/Sentiment Analyzer/Second-Preprocess.py'