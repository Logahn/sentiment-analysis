# Sentiment Analysis with Plotly and TextBlob

This code analyzes sentiment using TextBlob and the vaderSentiment libraries. It also uses the Plotly library to generate visualizations.

## Necessary Installs

This code requires the installation of the following packages:

- vaderSentiment
- plotly
- chart-studio

## Libraries

The following libraries are imported in this code:

- vaderSentiment
- plotly.subplots
- plotly.offline
- plotly.graph_objs
- cufflinks
- matplotlib.pyplot
- seaborn
- wordcloud
- TextBlob
- re
- nltk.sentiment.vader
- nltk
- pandas
- numpy

## Dataset

The dataset used in this code is imported from a CSV file and analyzed for missing values, duplicated values, and data types. The code also provides a function to check the number of unique values in each column of the dataset.

## Data Cleaning

The code performs data cleaning by removing non-alphabetic characters and converting all text to lowercase. It then uses TextBlob to compute the polarity and subjectivity of the review text.

## Sentiment Analysis

The sentiment of each review is then analyzed using vaderSentiment, and classified as positive, negative, or neutral. The code generates a countplot and a percentage plot to display the distribution of sentiment in the dataset.

## Link

Google colab file can be found [here](https://colab.research.google.com/drive/1Sqp9asPF5cRfOCf3oKTpIdlJ9Z4rJ1wE "Link to Colab File")
