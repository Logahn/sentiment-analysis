from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
from plotly.subplots import make_subplots
from plotly.offline import *
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import cufflinks as cf
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd
import numpy as np

"""### Library imports"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
init_notebook_mode(connected=True)
cf.go_offline()

warnings.filterwarnings("ignore")
warnings.warn("this will not show")

pd.set_option('display.max_columns', None)

"""### Dataset import"""

df = pd.read_csv('amazon.csv')

df.head()

df = df.sort_values("wilson_lower_bound", ascending=False)
df.drop('Unnamed: 0', inplace=True, axis=1)
df.head()


def missing_values_analysis(df):
    na_columns_ = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns_].isnull().sum().sort_values(ascending=True)
    ratio_ = (df[na_columns_].isnull().sum()/df.shape[0]
              * 100).sort_values(ascending=True)
    missing_df = pd.concat([n_miss, np.round(ratio_, 2)], axis=1, keys=[
                           'Missing Values', 'Ratio'])
    missing_df = pd.DataFrame(missing_df)
    return missing_df


def check_dataframe(df, head=5, tail=5):

    print("SHAPE".center(82, '~'))
    print('Rows: {}'.format(df.shape[0]))
    print('columns: {}'.format(df.shape[1]))
    print("TYPES".center(82, '~'))
    print(df.dtypes)
    print("".center(82, '~'))
    print(missing_values_analysis(df))
    print("DUPLICATED VALUES".center(83, '~'))
    print(df.duplicated().sum)
    print("QUANTITIES".center(82, '~'))
    print(df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_dataframe(df)


import pandas as pd

def check_class(dataframe):
    """
    Function that receives a dataframe and returns a new dataframe with the count of unique values for each column, 
    ordered by the number of unique values in descending order.
    
    Parameters:
    dataframe: pandas dataframe
    
    Returns:
    nunique_df: pandas dataframe
    """
    # Create a new dataframe with the count of unique values for each column
    nunique_df = pd.DataFrame({'Variable': dataframe.columns,
                               'Classes': [dataframe[i].nunique() for i in dataframe.columns]})
    
    # Sort the dataframe by the number of unique values in descending order
    nunique_df = nunique_df.sort_values('Classes', ascending=False)
    
    # Reset the index of the dataframe
    nunique_df = nunique_df.reset_index(drop=True)
    
    # Return the new dataframe
    return nunique_df


check_class(df)

constraints = ['#B34D22', '#EBE00C', '#1FEB0C', '#0C92EB', '#EB0CD5']


def categorical_variable_summary(df, column_name):
    # Create a plot with 1 row and 2 columns, with subplot titles for each column
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('Countplot', 'Percentage'), 
                        specs=[[{'type': 'xy'}, {'type': 'domain'}]])

    # Add a bar trace to the first subplot with the count of each category in the specified column
    fig.add_trace(go.Bar(y=df[column_name].value_counts().values.tolist(), 
                         x=[str(i) for i in df[column_name].value_counts().index], 
                         text=df[column_name].value_counts().values.tolist(), 
                         textfont=dict(size=14), 
                         name=column_name, 
                         textposition='auto', 
                         showlegend=True, 
                         marker=dict(color=constraints, line=dict(color='#DBE6EC', width=1))), 
                  row=1, col=1)
    
    # Add a pie trace to the second subplot with the percentage of each category in the specified column
    fig.add_trace(go.Pie(labels=df[column_name].value_counts().keys(), 
                         values=df[column_name].value_counts().values, 
                         textfont=dict(size=18), 
                         textposition='auto', 
                         showlegend=False, 
                         name=column_name, 
                         marker=dict(colors=constraints)), 
                  row=1, col=2)

    # Update the layout of the plot with the title and template
    fig.update_layout(title={'text': column_name, 
                             'y': 0.9, 
                             'x': 0.5, 
                             'xanchor': 'center', 
                             'yanchor': 'top'}, 
                      template='plotly_white')
    
    # Plot the figure
    iplot(fig)


categorical_variable_summary(df, 'overall')

"""### Data cleaning"""

df.reviewText.head()

review_example = df.reviewText[2031]
review_example

review_example = re.sub("[^a-zA-Z]", '', review_example)
review_example

review_example = review_example.lower().split()
review_example


def rt(x): return re.sub('[^a-zA-Z]', ' ', str(x))


df['reviewText'] = df['reviewText'].map(rt)
df['reviewText'] = df['reviewText'].str.lower()
df.head()

"""### Sentiment Analysis"""

df[['polarity', 'subjectivity']] = df['reviewText'].apply(
    lambda Text: pd.Series(TextBlob(Text).sentiment))

for index, row in df['reviewText'].iteritems():

    score = SentimentIntensityAnalyzer().polarity_scores(row)

    neg = score['neg']
    neu = score['neu']
    pos = score['pos']

    if neg > pos:
        df.loc[index, 'sentiment'] = "Negative"
    elif pos > neg:
        df.loc[index, 'sentiment'] = "Positive"
    else:
        df.loc[index, 'sentiment'] = "Neutral"

df[df['sentiment'] == 'Positive'].sort_values("wilson_lower_bound",
                                              ascending=False).head(5)

categorical_variable_summary(df, 'sentiment')
