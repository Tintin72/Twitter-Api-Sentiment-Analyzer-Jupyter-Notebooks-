import os
import tweepy
import ssl
import pandas as pd
import nltk 
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS


def TweetAnalyzer(tweeter_handle):
    """function to take  the most used words in a tweet"""
    plt.style.use('fivethirtyeight')

    # dotenv_path = join(dirname(__file__), '.env')
    load_dotenv()

    API_KEY = os.environ.get('TWITTER_API_KEY')

    API_SECRET_KEY = os.environ.get('TWITTER_API_SECRET')

    auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
    # auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET_TOKEN)

    api = tweepy.API(auth)

    tweets = []

    for page in range(1, 5):
        tweets.extend(api.user_timeline(
            screen_name=f'{tweeter_handle}', count=100,))
    print("number of tweets extracted will be {}".format(len(tweets)))

    own_tweets = [tweet for tweet in tweets if tweet.retweeted ==
                  False and 'RT @' not in tweet.text]

    df = pd.DataFrame(data=[[tweet.created_at, tweet.text, len(tweet.text), tweet.id, tweet.favorite_count, tweet.retweet_count] for tweet in own_tweets],
                      columns=['Date', 'Tweets', 'Length of text', 'id', 'Likes', 'Retweets'])
    df.head()

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('vader_lexicon')

    vader = SentimentIntensityAnalyzer()

    def f(tweet): return vader.polarity_scores(tweet)['compound']
    df['Sentiment'] = df['Tweets'].apply(f)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df.head()

    date_df = df.groupby(['Date']).mean().reset_index()
    # date_df.head()
    date_df.plot(kind='line', x="Date", y='Sentiment',
                 figsize=(20, 20), ylim=[-1, 1])
    plt.axhline(y=0, color='black')
    plt.xlabel('date')
    plt.ylabel(' sentiment of tweets')
    plt.title(
        f'Average sentiment analysis of {tweeter_handle} tweets against time')

    text = ' '.join(text for text in df.Tweets)

    stopwords = set(STOPWORDS)

    stopwords.update(['HTTPS', 'CO'])
    wordcloud = WordCloud(stopwords=stopwords,
                          background_color="white").generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    wordcloud.to_file("cloud.png")
    # plt.axis('off')
    # plt.savefig('sentiment.png')


if __name__ == '__main__':
    TweetAnalyzer("@WilliamsRuto")
