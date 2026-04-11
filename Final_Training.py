from Second_Preprocess import Embedding_Vector
from Cross_Validation import Bidirectional_Extended_RNNs
import joblib

max_token_each_input = 150
token_vector_dimension = 130

dataset = joblib.load ('setup_dataset.pkl')
token_vector_model = joblib.load ('tokenization_vectorization_model.pkl')
model = Bidirectional_Extended_RNNs (max_token_each_input, token_vector_dimension, 'gru')

train_review_text = dataset['review']
train_review_text = token_vector_model.X_setup (train_review_text, max_token_each_input)
train_ratings = dataset['star'].values

model.train (train_review_text, train_ratings, 128, 1)

joblib.dump (model, 'model.pkl')