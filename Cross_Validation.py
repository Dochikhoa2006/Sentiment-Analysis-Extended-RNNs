from Second_Preprocess import Embedding_Vector
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, GRU, Input, Masking, Bidirectional, Dense
from sklearn.model_selection import KFold
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib


class Bidirectional_Extended_RNNs ():

    def __init__ (self, max_token, dimension_token, name):

        self.max_token_each_input = max_token
        self.token_vector_dimension = dimension_token
        self.name = name
        self.model = self.create_model ()
    
    def construct_architecture (self):

        input_layer = Input (shape = (self.max_token_each_input, self.token_vector_dimension))
        input_layer_masked = Masking (mask_value = 0.0)(input_layer)

        if self.name == 'lstm':
            hidden_layer_1 = Bidirectional (LSTM (128, return_sequences = True))(input_layer_masked)
            hidden_layer_2 = Bidirectional (LSTM (64, return_sequences = False))(hidden_layer_1)
        else:
            hidden_layer_1 = Bidirectional (GRU (128, return_sequences = True))(input_layer_masked)
            hidden_layer_2 = Bidirectional (GRU (64, return_sequences = False))(hidden_layer_1)
        
        output_layer = Dense (5, activation = 'softmax')(hidden_layer_2)
    
        return input_layer, output_layer

    def create_model (self):

        input_layer, output_layer = self.construct_architecture ()
        self.model = Model (inputs = input_layer, outputs = output_layer)
        self.model.compile (optimizer = 'adam', loss = 'sparse_categorical_crossentropy')

        return self.model
    
    def train (self, X_train, Y_train, batch_size, epochs):
        
        self.model.fit (X_train, Y_train, epochs = epochs, batch_size = batch_size)

    def make_predictions (self, X_test):

        predictions = self.model.predict (X_test)
        predictions = np.argmax (predictions, axis = 1)

        return predictions
    
    def compute_accuracy_and_confusion_matrix (self, predictions, Y_test):
        
        accuracy = 0
        confusion_matrix = np.zeros ((5, 5), dtype = 'int64')

        for index in range (len (predictions)):
            predicted = np.int64 (predictions[index])
            observed = np.int64 (Y_test[index])

            confusion_matrix[predicted][observed] += 1
            if predictions[index] == Y_test[index]:
                accuracy += 1
        
        return np.float64 (accuracy / len (predictions)), confusion_matrix

def compute_mean_and_confidence_interval (accuracy):
    
    accuracy = np.array (accuracy)

    std_divide_sqrt = stats.sem (accuracy)
    inverse_CDF = stats.t.ppf ((1 + 0.94) / 2, len (accuracy) - 1)
    margin_size = std_divide_sqrt * inverse_CDF

    return np.mean (accuracy), margin_size

def plotting ():
    
    _, ((graph1, graph2), (graph3, graph4)) = plt.subplots (2, 2, figsize = (15, 8))

    sns.heatmap (confusion_matrix_BI_LSTM.T, annot = True, ax = graph1, xticklabels = [1, 2, 3, 4, 5], yticklabels = [1, 2, 3, 4, 5])
    graph1.set_title ("BI-LSTM Confusion Matrix")
    graph1.set_ylabel ('Predicted Ratings')
    graph1.set_xlabel ('Observed Ratings')

    sns.heatmap (confusion_matrix_BI_GRU.T, annot = True, ax = graph2, xticklabels = [1, 2, 3, 4, 5], yticklabels = [1, 2, 3, 4, 5])
    graph2.set_title ("BI-GRU Confusion Matrix")
    graph2.set_ylabel ('Predicted Ratings')
    graph2.set_xlabel ('Observed Ratings')

    graph3.errorbar (['BI-LSTM', 'BI-GRU'], mean, yerr = margin, fmt = 'o', color = 'blue', ecolor = 'red')
    graph3.set_title ('Confidence Interval (94%) of BI-LSTM & BI-GRU')
    graph3.set_ylabel ('Accuracy')

    graph4.axis ('off')

    plt.tight_layout ()
    plt.savefig ('Comparison_2_Models.png')
    plt.show ()


if __name__ == '__main__':
    
    dataset = joblib.load ('setup_dataset.pkl')
    token_vector_model = joblib.load ('tokenization_vectorization_model.pkl')

    cross_validation = KFold (n_splits = 10, shuffle = True, random_state = 100)
    train_test_split = cross_validation.split (dataset)

    max_token_each_input = 150
    token_vector_dimension = 130

    accuracy_BI_LSTM = []
    accuracy_BI_GRU = []
    confusion_matrix_BI_LSTM = np.zeros ((5, 5), dtype = 'int64')
    confusion_matrix_BI_GRU = np.zeros ((5, 5), dtype = 'int64')


    for train_index, test_index in train_test_split:

        train_dataset = dataset.iloc[train_index]
        test_dataset = dataset.iloc[test_index]

        train_review_text = train_dataset['review']
        train_review_text = token_vector_model.X_setup (train_review_text, max_token_each_input)
        train_ratings = train_dataset['star'].values

        test_review_text = test_dataset['review']
        test_review_text = token_vector_model.X_setup (test_review_text, max_token_each_input)
        test_ratings = test_dataset['star'].values

        BI_LSTM = Bidirectional_Extended_RNNs (max_token_each_input, token_vector_dimension, 'lstm')
        BI_LSTM.train (train_review_text, train_ratings, 128, 1)
        predictions_BI_LSTM = BI_LSTM.make_predictions (test_review_text)
        a, b = BI_LSTM.compute_accuracy_and_confusion_matrix (predictions_BI_LSTM, test_ratings)
        accuracy_BI_LSTM.append (a)
        confusion_matrix_BI_LSTM += b

        BI_GRU = Bidirectional_Extended_RNNs (max_token_each_input, token_vector_dimension, 'gru')
        BI_GRU.train (train_review_text, train_ratings, 128, 1)
        predictions_BI_GRU = BI_GRU.make_predictions (test_review_text)
        c, d = BI_GRU.compute_accuracy_and_confusion_matrix (predictions_BI_GRU, test_ratings)
        accuracy_BI_GRU.append (c)
        confusion_matrix_BI_GRU += d


    mean_BI_LSTM, margin_BI_LTSM = compute_mean_and_confidence_interval (accuracy_BI_LSTM)
    mean_BI_GRU, margin_BI_GRU = compute_mean_and_confidence_interval (accuracy_BI_GRU)
    mean = [mean_BI_LSTM, mean_BI_GRU]
    margin = [margin_BI_LTSM, margin_BI_GRU]

    plotting ()





# cd '/Users/chikhoado/Desktop/PROJECTS/Sentiment Analyzer'
# /opt/homebrew/bin/python3.12 -m venv .venv
# source .venv/bin/activate
# pip install --upgrade pip
# brew install openjdk@17 apache-spark
# sudo ln -sfn /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-17.jdk
# pip install numpy tensorflow scikit-learn scipy matplotlib joblib
# python '/Users/chikhoado/Desktop/PROJECTS/Sentiment Analyzer/Cross_Validation.py'