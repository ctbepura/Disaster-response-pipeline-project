import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    Takes input rom two csv files and merges them into a single pandas dataframe
    
    Input:
    messages_filepath = filepath to messages csv file
    categories_filepath = filepath to categories csv file
    
    Returns:
    df dataframe merging the two files    
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #merge datasets
    return messages.merge(categories, on = "id")
    
    
def clean_data(df):
    '''
    Takes a df and cleans it
    
    Input:
    df = the merged data frame from the previous step
    
    Returns:
    df = the data frame after cleaning     
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat = ';', expand = True)
    
    # select the first row of the categories dataframe
    row = categories[0:1][:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = list(row.squeeze().str.split('-').str.get(0))
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].squeeze().str.split('-').str.get(1)
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    
    # drop the original categories column from `df`
    df = df.drop('categories', axis =1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df
    

def save_data(df, database_filename):
    '''
    save-data
    Saves the cleaned data frame in a sqlite database
    
    Input:
    df = cleaned data frame
    database_filename = name and path to the sqlite database
    
    Result
    df is saved to sqlite database
    '''
    
    #Save the dataframe in a sqlite database
    connect_str = f"sqlite:///{database_filename}"
    engine = create_engine(connect_str)
    
    df.to_sql("messages", engine, if_exists='replace', index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
        
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()