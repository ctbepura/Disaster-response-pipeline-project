# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

### General background and Overview:
This project was done in partial fulfillment of the Data scientist nanodegree program at Udacity. The project analyses real-world data from messages that were sent during natural disasters. Through a process of Data cleaning, ETL and Machine Learning through Natural Language processing, the project builds a model that can be used during future natural disasters to quickly process the messages being received, correctly categorise them and allocate them to the relevant support teams. This project therefore has potential to solve a real world need.

### Structure of the key files in the repository
![image](https://user-images.githubusercontent.com/46715348/170452897-0128c237-d039-4dd1-9bdc-0036cd25e4e9.png)



### Acknowledgements
I would like to thank Appen (formally Figure 8) for providing the data as well as Udacity and my mentors at Udacity for the help throughout the project,
