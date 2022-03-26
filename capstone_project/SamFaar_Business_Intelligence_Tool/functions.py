#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')
mpl.rcParams['figure.figsize'] = (9, 6)
sns.set_context('talk', font_scale=1)

from finpie import NewsData
from bs4 import BeautifulSoup as soup
from urllib.request import Request, urlopen
import re

import datetime as dt
from datetime import date, timedelta
from newsapi.newsapi_client import NewsApiClient
import nltk
nltk.download('vader_lexicon')
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
from wordcloud import WordCloud, STOPWORDS

# pre-process: removing stopwords, stemming, lemmatization
import gensim
from gensim.utils import simple_preprocess   # Convert a document into a list of lowercase tokens, ignoring tokens that are too short or too long.
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer


def get_news(company_ticker, search_date):
    ## newsapi
    newsapi = NewsApiClient(api_key='3a2d0a55066041dc81e3acfbd665fc6e')
    # gives from search_date to today's date
    articles = newsapi.get_everything(q=company_ticker,           # newsapi works pretty well with only the ticker, so we don't add comapny name
                                      from_param=search_date,
                                      language="en",
                                      sort_by="publishedAt",
                                      page_size=100)
    df_newsapi = pd.DataFrame(articles['articles'])
    # do some cleaning of the DF
    df_newsapi.drop(['author', 'urlToImage'], axis=1, inplace=True)
    df_newsapi.rename({'publishedAt': 'datetime'}, axis=1, inplace = True)
    df_newsapi.rename({'title': 'news_headline'}, axis=1, inplace = True)
    df_newsapi['source'] =  df_newsapi['source'].map(lambda x: x['name'])
    
    
    ## finviz
    url = ("http://finviz.com/quote.ashx?t=" + company_ticker.lower())
    # Most websites block requests that are without a User-Agent header (these simulate a typical browser)
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    html = soup(webpage, "html.parser")
    news = pd.read_html(str(html), attrs={'class':
                                          'fullview-news-outer'})[0]
    links = []
    for a in html.find_all('a', class_="tab-link-news"):
        links.append(a['href'])
    # Clean up news dataframe
    news.columns = ['Date', 'News_Headline']
    news['Article_Link'] = links
    # >>> clean "Date" column and create a new "datetime" column
    # extract time
    news['time'] = news['Date'].apply(lambda x: ''.join(re.findall(r'[a-zA-Z]{1,9}-\d{1,2}-\d{1,2}\s(.+)', x)))
    news.loc[news['time'] == '', 'time'] = news['Date']
    #extract date
    news['date'] = news['Date'].apply(lambda x: ''.join(re.findall(r'([a-zA-Z]{1,9}-\d{1,2}-\d{1,2})\s.+', x)))
    news.loc[news['date'] == '', 'date'] = np.nan
    news.fillna(method = 'ffill', inplace = True)
    # convert to datetime type
    news['datetime'] = pd.to_datetime(news['date'] + ' ' + news['time'])
    news.drop(['Date', 'time', 'date'], axis = 1, inplace = True)
    news.sort_values('datetime', inplace = True)
    news.reset_index(drop=True, inplace =True)
    news.columns = ['news_headline', 'url', 'datetime']
    df_finviz = news.copy()
    
    ## GoogleNews
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
    config = Config()
    config.browser_user_agent = user_agent
    config.request_timeout = 10
    # GoogleNews sometime returns an empy dataframe, so we add a while to force GoogleNews to retun a result 
    df_google = pd.DataFrame()
    while df_google.shape[0] == 0:
        # Extract News with Google News ---> gives only 10 results per request
        googlenews = GoogleNews(start=date.today())
        googlenews.search(company_ticker)
        result = googlenews.result()
        # store the results
        df_google = pd.DataFrame(result)
        # do some cleaning of the DF
        if df_google.shape[0] != 0:
            df_google.drop(['img', 'date'], axis=1, inplace=True)
            df_google.columns = ['news_headline', 'source', 'datetime', 'description', 'url']
    df_google
    
    ## Add all three DFs together
    df_news = pd.concat([df_newsapi, df_finviz, df_google], ignore_index=True)
    df_news['datetime'] = pd.to_datetime(df_news['datetime'], format = '%Y-%m-%d %H:%M:%S')
    df_news.set_index('datetime', inplace = True)
    # only returning the rows that match our search_date
    df_news = df_news[df_news.index.to_period('D') == search_date]
    df_news.sort_index(inplace = True)
    # Get clean source column from urls using regex
    df_news['source'] = df_news['url'].map(lambda x: ''.join(re.findall(r"https?://(?:www.)?([A-Za-z_0-9.-]+).*", x)))
    
    return df_news



def nltk_vader_score(text):
    sentiment_analyzer = SIA()
    # we take "compound score" (from -1 to 1): The normalized compound score which calculates the sum of all lexicon ratings
    sent_score = sentiment_analyzer.polarity_scores(text)['compound']
    return sent_score



def sentiment_type(text):
    analyzer = SIA().polarity_scores(text)
    neg = analyzer['neg']
    neu = analyzer['neu']
    pos = analyzer['pos']
    comp = analyzer['compound']
    
    if neg > pos:
        return 'negative'
    elif pos > neg:
        return 'positive'
    elif pos == neg:
        return 'neutral'



def tokenizer(text):
    import nltk

    #     nltk.download('stopwords')
    from nltk.corpus import stopwords

    new_stopwords = [
        'monkey', 'wall', 'zacks', 'motley', 'fool', 'theStreet.com', 'yahoo',
        'finance', 'video', "investor's", 'business', 'daily', 'globeNewswire',
        'stock', 'stocks', 'company', 'invest', 'investing', 'today',
        'according'
    ]

    stpwrd = nltk.corpus.stopwords.words('english')
    stpwrd.extend(new_stopwords)

    #     nltk.download('punkt')
    from nltk.tokenize import word_tokenize

    text_tokens = word_tokenize(text)

    return ', '.join(
        [words for words in text_tokens if words.lower() not in stpwrd])



def word_cloud(text):
    stopwords = set(STOPWORDS)
    allWords = ' '.join([nws for nws in text])
    wordCloud = WordCloud(
        background_color='white',  # black
        width=1600,
        height=800,
        stopwords=stopwords,
        min_font_size=20,
        max_font_size=150).generate(allWords)
    return wordCloud


    


def get_article_text(Article_Link):
    import requests
    from bs4 import BeautifulSoup

    # using request package to make a GET request for the website, which means we're getting data from it.
    header = {
        "User-Agent":
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest"
    }

    html = requests.get(Article_Link, headers=header).content
    soup = BeautifulSoup(html)

    # Get the whole body tag
    tag = soup.body

    # Join each string recursively
    text = []
    for string in tag.strings:
        # ignore if fewer than 15 words
        if len(string.split()) > 15:
            text.append(string)
    return ' '.join(text)



def keyword_extractor(text):
    from flashtext import KeywordProcessor
    kwp = KeywordProcessor()

    keyword_dict = {
        'new product': ['new product', 'new products'],
        'M&A': ['merger', 'acquisition'],
        'stock split/buyback': ['buyback', 'split'],
        'workforce change': ['hire', 'hiring', 'firing', 'lay off', 'laid off']
    }

    kwp.add_keywords_from_dict(keyword_dict)
    
    # we use set to get rid of repeating keywords, and ', '.join() to get string instead of SET data type:
    return ', '.join(set(kwp.extract_keywords(text)))

