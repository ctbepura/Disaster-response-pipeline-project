import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

import re
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn import metrics
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    '''
    load_data
    connects to the sqlite data base,  loads the data and splits the X and Y data
    
    Input:
    database_filepath = the sqlite database filepath where data is being loaded from
    
    Return:
    X = the input variables for the ML algorithm
    Y = the output varibale for the ML algorithm
    category_names = the message classification categories    
    '''    
    connect_str = f"sqlite:///{database_filepath}"
    engine = create_engine(connect_str)
    df = pd.read_sql("SELECT * FROM messages", engine)
    df.related.unique()
    df = df[df.related != 2]
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = Y.columns.tolist()
    
    return X, Y, category_names


def tokenize(text):
    '''
    tokenize:
    Takes a message string and splits it into clean tokens
    
    Input:
    text = the raw message tha twas received
    
    Returns:
    clean_tokens = the tokenized text after normalizing, lemmatizing and cleaning the text    
    '''
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok, pos = 'v').lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    build_model
    defines the pipeline / grid search object that will be used as the ML model and creates a Grid Search object
    
    Returns
    The grid search object
    '''
    pipeline = Pipeline ([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
              ])
    
    #Specify parameters for the grid search
    parameters = {
        #"features_text_pipeline_vect_ngram_range": ((1, 1), (1, 2)),
        #"clf_estimator_min_samples_split": [2, 4],
        
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [2, 4],
        }
    
    #create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv
    
    


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model
    scores and prints the performance of the model in predicting the message classifications    
    '''
    
    print()
    
    y_pred = model.predict(X_test)
    
    print(metrics.classification_report(Y_test, y_pred, target_names = category_names))
    print()



def save_model(model, model_filepath):
    '''
    save_model
    saves the model as a pickle file    
    '''
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()