from Second_Preprocess import Embedding_Vector
from Cross_Validation import Bidirectional_Extended_RNNs
import numpy as np
import joblib

max_token_each_input = 150
token_vector_dimension = 130

token_vector_model = joblib.load ('tokenization_vectorization_model.pkl')
model = joblib.load ('model.pkl')

print ('\n+---------------------------------------------------------------------------------+')
print ('-> Give a review about your experience with our application on the Amazon Appstore')
review = input ('Your review: ')
print ('-> The system is working, please wait...')
processed_review = token_vector_model.X_setup ([review], max_token_each_input)
sentiment_val = model.make_predictions (processed_review)
sentiment_val = np.argmax (sentiment_val)

if sentiment_val == 0:
    sentiment_text = 'strongly dissatisfied'
elif sentiment_val == 1:
    sentiment_text = 'dissatisfied'
elif sentiment_val == 2:
    sentiment_text = 'neutral'
elif sentiment_val == 3:
    sentiment_text = 'satisfied'
else:
    sentiment_text = 'strongly satisfied'

print (f'-> This review shows a {sentiment_text} feeling with {sentiment_val + 1}/5 app star !!')