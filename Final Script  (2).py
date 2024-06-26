#!/usr/bin/env python
# coding: utf-8

# In[68]:


#Imports and Installs

import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt 
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models.coherencemodel import CoherenceModel
get_ipython().system('pip install pyLDAvis')
import pyLDAvis.gensim_models
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
import numpy as np



# Step 2: Load the CSV file
#df = pd.read_csv('covid19_tweets.csv')
df = pd.read_csv('covid19_tweets.csv')

# Step 3: Examine the data

print(df.info())  # Provides a concise summary of the DataFrame

#From the df I create a dataframe to contain the columns# Subset and rename columns in one step
ss_df2 = df[['user_location', 'date', 'text', 'hashtags']].rename(columns={
    'user_location': 'location',
    'date': 'date',
    'text': 'text',
    'hashtags': 'hashtags'
})

# Get the first and last date
first_date1 = df['date'].min()
last_date1 = df['date'].max()
distinct_locations = ss_df2['location'].unique()

# Print the results
print("First date in the dataset:", first_date1)
print("Last date in the dataset:", last_date1)
print(distinct_locations)


# In[69]:


distinct_locations = ss_df2['location'].dropna().unique()
distinct_locations.sort()
with pd.option_context('display.max_rows', None):
    print(distinct_locations)


# In[70]:


#Filtering info & defining keywords
# Define the keywords to search for
keywords = ['supply chain', 'logistics', 'delivery', 'freight', 'out of stock', 'distribution', 'warehousing', 'warehouse management system', 'shipping order', 'transport'
           'carrier', 'shipment', 'shipping', 'tracking number', 'SCM', 'supply network', 'import', 'export', 'trade', 'tarrif', 'supply', 'demand', 'optimization']

# Filter rows where the text column contains any of the keywords
# 
filtered_df = ss_df2[ss_df2['text'].str.contains('|'.join(keywords), case=False, na=False)]
filt_df_hastag = ss_df2[ss_df2['hashtags'].str.contains('|'.join(keywords), case=False, na=False)]
print(filtered_df.info())
print(filtered_df.head(100))
print(filt_df_hastag.info())


# In[71]:




# Initialize the lemmatizer
word_lemmatizer = WordNetLemmatizer()

# Function to tokenize and lemmatize text
def process_text(text):
    # Tokenizing the text
    tokenized = word_tokenize(text)
    # Lemmatizing each token
    return [word_lemmatizer.lemmatize(token) for token in tokenized]

# Construct a list of stopwords including punctuation and numbers
base_stopwords = stopwords.words('english') + list(string.punctuation) + [str(i) for i in range(10)]
additional_stopwords = ['’', "'s", '...', '``', "n't", '…', "''", '“']
combined_stopwords = base_stopwords + additional_stopwords

# Function to tokenize text and remove stopwords
def filter_tokens(text):
    # Tokenizing the text
    tokens = word_tokenize(text)
    # Filtering tokens by stopwords
    return [token.lower() for token in tokens if token.lower() not in combined_stopwords]

# Function to remove URLs from dataframe
def clear_urls(df):
    df['text'] = df['text'].str.replace(r"http\S+", "", regex=True)

# Helper function for lemmatization
def lemmatize_tokens(tokens):
    return [word_lemmatizer.lemmatize(token) for token in tokens]


# In[72]:


#applying functions
dataset = filtered_df
clear_urls(dataset)
dataset['text']=dataset['text'].apply(filter_tokens)

# Debugging output
print(dataset['text'].head())

dataset['text'] = dataset['text'].apply(lemmatize_tokens)
all_words2 = [word for tokens in dataset['text'] for word in tokens]
tweet_lengths2 = [len(tokens) for tokens in dataset['text']]
vocab = sorted(list(set(all_words2)))
print('{} words total, with a vocabulary size of {}'.format(len(all_words2), len(vocab)))
print('Max tweet length is {}'.format(max(tweet_lengths2)))


plt.figure(figsize = (15,8))
sns.countplot(tweet_lengths2)
plt.title('Tweet Length Distribution', fontsize = 18)
plt.xlabel('Words per Tweet', fontsize = 12)
plt.ylabel('Number of Tweets', fontsize = 12)
plt.show()

#Build wordcloud to see distirbution

from nltk.probability import FreqDist

#iterate through each tweet, then each token in each tweet, and store in one list
flat_words = [item for sublist in dataset['text'] for item in sublist]

word_freq = FreqDist(flat_words)

word_freq.most_common(10)


# In[73]:


#Next deduce a bag of words Using GenSims Dictionary to deduce each

#create dictionary
text_dict = Dictionary((dataset['text']))

#text_dict = {word: 1 for word in vocab}
#view integer mappings
text_dict.token2id

tweets_bow = [text_dict.doc2bow(tweet) for tweet in dataset['text']]

#Now I move onto fitting the Latent Dirchlet Allocation, a popular topic modelling algorithms

k = 5
tweets_lda = LdaModel(tweets_bow,
                      num_topics = k,
                      id2word = text_dict,
                      random_state = 100,
                      chunksize=100,
                      passes=10)

tweets_lda.show_topics()
#VIsualization on topics

#now I perform a visualization on the topic mode with pyLDAvis

#use pyLDAv is to show the topic modelling of the SC related tweets in twitter.
pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim_models.prepare(tweets_lda, tweets_bow, dictionary=tweets_lda.id2word, n_jobs=1)
vis


# In[ ]:





# In[74]:


# If 'filtered_df' is a slice from another DataFrame, make a copy to avoid SettingWithCopyWarning:
dataset = filtered_df.copy()

# Now apply changes
dataset['date'] = pd.to_datetime(dataset['date'])

# Check the data type of the 'date' column to ensure it's datetime
print(dataset['date'].dtype)

# If it's not datetime, convert it
dataset['date'] = pd.to_datetime(dataset['date'])

# Now, let's retry grouping
try:
    topic_trends = dataset.groupby([pd.Grouper(key='date', freq='W'), 'topic']).size().unstack()
    print(topic_trends.head())
except Exception as e:
    print("An error occurred:", e)
    
# Assuming your dataset is prepared and the 'text' column contains cleaned and tokenized text
dataset['tokens'] = dataset['text'].apply(lambda x: text_dict.doc2bow(x))

# Function to assign the most probable topic to each tweet's tokens
def assign_topic(tokens):
    topics = tweets_lda.get_document_topics(tokens)
    return max(topics, key=lambda x: x[1])[0]+1  # Return the topic number with the highest probability

# Apply the function to assign topics
dataset['topic'] = dataset['tokens'].apply(assign_topic)

# Group by date and topic, counting occurrences
topic_trends = dataset.groupby([pd.Grouper(key='date', freq='W'), 'topic']).size().unstack()

# Plot the trends over time for each topic
topic_trends.plot(figsize=(15, 7), subplots=True, layout=(-1, 3), sharex=True, title="Weekly Topic Trends")
plt.show()


# In[75]:


# Get the first and last date
first_date = dataset['date'].min()
last_date = dataset['date'].max()

# Print the results
print("First date in the dataset:", first_date)
print("Last date in the dataset:", last_date)


# In[76]:


# Define the topics and their associated keywords
topics = {
    'Topic 1 - Global Trade and COVID Impact': ['COVID', 'supply', 'imported', 'chain', 'trade', 'global', 'demand'],
    'Topic 2 - Public Health and COVID': ['COVID19', 'health', 'people'],
    "Topic 3 - COVID's Economic Impact": ['Demand', 'COVID', 'delivery', 'trade'],
    'Topic 4 - COVID and Lockdown Effects': ['Demand', 'importance', 'COVID', 'lockdown', 'service'],
    'Topic 5 - Supply Chain Dynamics': ['Distribution', 'shipping', 'supplying', 'important', 'COVID19']
}

# Create a figure and a grid of subplots
fig, axs = plt.subplots(3, 2, figsize=(10, 15), dpi=100)

# Remove empty subplot (if any)
fig.delaxes(axs[2, 1])

# Colors for each subplot
colors = ['skyblue', 'lightgreen', 'salmon', 'khaki', 'lightgrey']
border_colors = ['blue', 'green', 'red', 'darkkhaki', 'grey']

for ax, (title, keywords), bg_color, border_color in zip(axs.flat, topics.items(), colors, border_colors):
    ax.set_title(title, fontsize=14, color='black')
    ax.set_axis_off()
    ax.set_facecolor(bg_color)

    # Add border color
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor(border_color)

    # Join keywords into a multi-line string, add "etc." and place them in the square
    text = '\n'.join(keywords) + '\netc.'
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12, color='black')

plt.tight_layout()
plt.show()


# In[77]:


# Now perform sentiment analysis across the topics
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to get sentiment
def get_sentiment(text):
    # VADER outputs sentiment scores for positive, negative, neutral, and an overall score
    scores = sid.polarity_scores(text)
    return scores['compound']  # Returning the compound score as an overall measure of sentiment

# Apply sentiment analysis to the original text
dataset['sentiment'] = dataset['text'].apply(lambda x: ' '.join(x))  # Join tokens to form strings
dataset['sentiment_score'] = dataset['sentiment'].apply(get_sentiment)


# In[78]:


# Group by topic and calculate mean sentiment score
average_sentiment_per_topic = dataset.groupby('topic')['sentiment_score'].mean()
print(average_sentiment_per_topic)


# In[79]:


import seaborn as sns

# Plotting sentiment distribution for each topic
plt.figure(figsize=(10, 6))
sns.boxplot(x='topic', y='sentiment_score', data=dataset)
plt.title('Sentiment Score Distribution by Topic')
plt.xlabel('Topic')
plt.ylabel('Sentiment Score')
plt.show()




# In[ ]:





# In[126]:


# Load the data
BADIdf = pd.read_csv('Baltic Dry Index Historical Data.csv')
BADIdf['Date'] = pd.to_datetime(BADIdf['Date'])
BADIdf['Price'] = BADIdf['Price'].str.replace(',', '').astype(float)

# Rename columns for consistency
BADI = BADIdf[['Date', 'Price']].rename(columns={'Date': 'date'})

# Set 'date' as the index for resampling
BADI.set_index('date', inplace=True)
BADI_weekly = BADI.resample('W-SUN').mean().reset_index()  # Summarize weekly and reset index

print(BADI_weekly.head())


# In[127]:


# Assuming topic_trends loaded similarly and needs 'date' as a column
if 'date' not in topic_trends.columns:
    topic_trends.reset_index(inplace=True)

topic_trends['date'] = pd.to_datetime(topic_trends['date'])  # Ensure 'date' is in datetime format

print(topic_trends.head())


# In[128]:


merged_df = pd.merge(BADI_weekly, topic_trends, on='date', how='inner')
print(merged_df.head())


# In[129]:


print(merged_df.info())


# In[ ]:





# In[131]:


import matplotlib.pyplot as plt

# Plotting setup
fig, axes = plt.subplots(nrows=len(topic_trends.columns) // 3 + 1, ncols=3, figsize=(15, 8), sharex=True)  # Adjusted figsize
axes = axes.flatten()  # Flatten the axes array for easier iteration

# Iterate over each topic to create a subplot
for i, topic in enumerate(topic_trends.columns):
    # Plot topic trend on the primary y-axis
    axes[i].plot(merged_df.index, merged_df[topic], label=f"Topic {topic}", color='tab:blue')
    axes[i].set_title(f"Topic {topic} Trend", fontsize=10)
    axes[i].set_ylabel("Occurrences")
    axes[i].legend(loc='upper left')

    # Plot BDI price on the secondary y-axis
    ax2 = axes[i].twinx()
    ax2.plot(merged_df.index, merged_df['Price'], label="BDI Price", color='tab:red', linestyle='--')
    ax2.set_ylabel("BDI Price")
    ax2.legend(loc='upper right')

# Adjust layout
plt.tight_layout(pad=3.0, h_pad=1.0, w_pad=1.0)
fig.subplots_adjust(top=0.85)  # Adjust the top to make space for the super title
fig.suptitle("Weekly Topic Trends with BDI Overlay", fontsize=16, y=0.98)

# Hide unused subplots if any
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.show()


# In[ ]:





# In[135]:


#Correlation and Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# Correlation analysis
correlation_matrix = merged_df[['Price', 1, 2, 3, 4, 5]].corr()
print("Correlation matrix between Price and Topics:")
print(correlation_matrix.loc['Price'])

# Regression analysis
for topic in [ 1, 2, 3, 4, 5]:
    X = merged_df[[topic]]  # Predictor variable (each topic)
    y = merged_df['Price']  # Response variable (BDI Price)

    model = LinearRegression()
    model.fit(X, y)

    # Predicting and checking performance
    y_pred = model.predict(X)
    print(f"Regression analysis for Topic {topic}:")
    print(f"  Coefficient: {model.coef_[0]:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")
    print(f"  R-squared score: {r2_score(y, y_pred):.4f}\n")


# In[133]:


print(merged_df.columns)


# In[ ]:





# In[ ]:




