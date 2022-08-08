import os
import tweepy
import ssl
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS


def TweetAnalyzer(twitter_handle):
    """function to take  the most used words in a tweet"""
    # declaring fivethirtyeight as the styling for graphs
    plt.style.use('fivethirtyeight')

    # load env variables
    load_dotenv()

    API_KEY = os.environ.get('TWITTER_API_KEY')

    API_SECRET_KEY = os.environ.get('TWITTER_API_SECRET')

    # authenticate using tweepy passing api key & secret key
    auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)

    api = tweepy.API(auth)

    tweets = []

    # obtain the last 400 tweets from the twitter handle
    for page in range(1, 5):
        tweets.extend(api.user_timeline(
            screen_name=f'{twitter_handle}', count=100,))
    print("number of tweets extracted will be {}".format(len(tweets)))

    # remove all retweets to obtain original tweets
    own_tweets = [tweet for tweet in tweets if tweet.retweeted ==
                  False and 'RT @' not in tweet.text]

    df = pd.DataFrame(data=[[tweet.created_at, tweet.text, len(tweet.text), tweet.id, tweet.favorite_count, tweet.retweet_count] for tweet in own_tweets],
                      columns=['Date', 'Tweets', 'Length of text', 'id', 'Likes', 'Retweets'])
    df.head()  # head method to view the first few records in the table

    # bypass ssl errors in downloading vader_lexicon
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('vader_lexicon')

    # instantiate vader
    vader = SentimentIntensityAnalyzer()

    def f(tweet): return vader.polarity_scores(tweet)[
        'compound']  # function to perform sentiment analysis
    df['Sentiment'] = df['Tweets'].apply(f)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df.head()

    """
    sort the records by date and create a data frame 
    offsetting from the mean value obtained from the sentiment score
    """
    date_df = df.groupby(['Date']).mean().reset_index()
    date_df.plot(kind='line', x="Date", y='Sentiment',
                 figsize=(20, 20), ylim=[-1, 1])
    plt.axhline(y=0, color='black')
    plt.xlabel('date')
    plt.ylabel(' sentiment of tweets')
    plt.title(
        f'Average sentiment analysis of {twitter_handle} tweets against time')

    text = ' '.join(text for text in df.Tweets)

    """
    remove unwanted words using stopwords 
    including words like https which occure in tweets
    """
    stopwords = set(STOPWORDS)

    stopwords.update(['HTTPS', 'CO'])
    wordcloud = WordCloud(stopwords=stopwords,
                          background_color="white").generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    #create an image from the generated words
    wordcloud.to_file(f"{twitter_handle}.png")

