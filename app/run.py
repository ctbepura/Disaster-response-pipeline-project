import sys
import json
import plotly
import pickle
import pandas as pd
import re
import sqlite3

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)


# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #Second graph data
    col_names = df.columns.tolist()[4:]
    df_1 = df.append(df.sum().rename('Total'))
    class_counts = df_1[-1:][col_names].mean().sort_values(ascending = False).head(20)
    class_names = df_1[-1:][col_names].mean().sort_values(ascending = False).head(20).index
    
    #3rd graph data
    df['total_column'] = df[col_names].sum(axis=1)
    class_per_message_counts = df.groupby('total_column').count()['message'].sort_values(ascending = False)
    class_per_message_names = class_per_message_counts.index
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
     
        
        
        {
            'data': [
                Bar(
                    x=class_names,
                    y=class_counts
                )
            ],

            'layout': {
                'title': 'Distribution of top 20 Message Classifications',
                'yaxis': {
                    'title': "Count of messages"
                },
                'xaxis': {
                    'title': "Message Classifications"
                }
            }
        },
      
        
        
        {
            'data': [
                Bar(
                    x =class_per_message_counts.index ,
                    y=class_per_message_counts
                   
                    
                )
            ],

            'layout': {
                'title': 'Messages grouped by classifications per message',
                'yaxis': {
                    'title': "Count of messages"
                },
                'xaxis': {
                    'title': "Count of classifications"
                }
            }
        }
        
        
    ]
      
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()