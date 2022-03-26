#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime
import streamlit as st
from st_aggrid import AgGrid   # for nicer DF and adjustable DF column size / pip install streamlit-aggrid
from functions import *
from functions import get_news, nltk_vader_score, sentiment_type, tokenizer, word_cloud, get_article_text, keyword_extractor

# st.set_page_config(page_title="", layout="wide")

# main sections of our App
header = st.container()
import_data = st.container()
news_headline_SA = st.container()
news_headline_word_cld = st.container()
kwrd_extract = st.container()


with header:
    st.title('Business Intelligence Gathering Tool for Financial Events & Sentiment Analysis using NLP')
    
    """
    * This tool will mine the web via three APIs (NEWS API, FINVIZ, and GoogleNews) to extract and gather financial news about a target company on a specific day.
      * **Tools used**: _urllib.request, BeautifulSoup, regex_
    * It will then do sentiment analysis on news headlines to predict the tone (percent of positive, negative, and neutral).
      * **Tools used**: _nltk.sentiment.vader, gensim.parsing.preprocessing, nltk.stem_
    * Lastly, it will extract important company events from the news texts that could impact their stock price, e.g. new products to launch, M&A, stock buybacks or splits, increase or decrease in hiring."
      * **Tools used**: _KeywordProcessor from flashtext library_
    """
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    
    
with import_data:
    st.header('Web Scraping via NEWS API, FINVIZ, and GoogleNews APIs')
    
    ticker_col, date_col = st.columns(2)
    
#     company_ticker = ticker_col.text_input('Company Ticker: ', 'AMZN')
    company_ticker = ticker_col.selectbox('Select Company Ticker', options = ['AMZN', 'AAPL', 'GOOG', 'MSFT', 'NVDA', 'Other'], index = 0)
    if company_ticker == 'Other':
        company_ticker = ticker_col.text_input('Type Company Ticker below', 'TSLA')
        
    search_date = date_col.date_input(
                                    'Select Date for Web News Extraction',
                                    datetime.date.today() - datetime.timedelta(days=1),  # default date
                                    max_value = datetime.date.today(),
                                    min_value=datetime.date.today() - datetime.timedelta(days=4)).strftime('%Y-%m-%d')   # for now, we can get news up to 4 days ago from today
    
    df_news = get_news(company_ticker, search_date)
    st.subheader(f"{df_news.shape[0]} news articles were retrieved from {df_news['source'].nunique()} news platforms")   # "###" is for markdown: st.makrdown("### Key Metrics")
    st.text("Hover over each column to see the 3-line menu icon for filtering")
    AgGrid(df_news[['source', 'news_headline', 'url']])
    
    # count per news platform
    st.subheader('Number of news articles extracted from each online news platform')
    fig = px.histogram(df_news, x='source', color = 'source').update_xaxes(categoryorder="total descending")
    fig.update_layout(xaxis_title='', font=dict(size=16), width=810, height=520, showlegend=False)
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
   
    
with news_headline_SA:
    st.header('Sentiment Analysis on News Headlines via NLTK VADER')
    df_news['sentiment_score_vader'] = df_news['news_headline'].map(nltk_vader_score)
    
    # Histogram of SA scores
    st.subheader("Histogram of News Headlines Sentiment Scores for " + company_ticker + " on " + search_date)  
    fig = px.histogram(df_news, x='sentiment_score_vader', color = 'source').update_xaxes(categoryorder="total descending")
    fig.update_layout(xaxis_title='Sentiment Score (Compound from -1 to 1)', font=dict(size=16), width=790, height=520, showlegend=False)
    st.plotly_chart(fig)
    
    
    # sentiment type
    df_news['sentiment_type'] = df_news['news_headline'].map(sentiment_type)
    st.subheader("Sentiment Type for each News")
    st.text("Hover over each column to see the 3-line menu icon for filtering")
    AgGrid(df_news[['sentiment_type', 'source', 'news_headline', 'url']])
    
    # pie chart of entiment type
    st.subheader("Percent of Sentiment Type from all News Headlines for " + company_ticker + " on " + search_date)
    fig = px.pie(df_news,
                 values=df_news['sentiment_type'].value_counts(normalize=True) * 100,
                 names=df_news['sentiment_type'].unique(),
                 color=df_news['sentiment_type'].unique(),
                 hole=0.35,
                 color_discrete_map={
                     'neutral': 'silver',
                     'positive': 'mediumspringgreen',
                     'negative': 'orangered'
                 })
    fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=22, hoverinfo='label+value', 
                      texttemplate = "%{label}<br>%{value:.0f}%")   # texttemplate = "%{text}<br>(%{a:.2f}, %{b:.2f}, %{c:.2f})"
    fig.update_layout(font=dict(size=16),
                      width=750,
                      height=520)
    st.plotly_chart(fig)
    
    
    # count of sentiment type per news platform
    st.subheader('Sentiment type distribution for each online news platform')
    # count per news platform with sentiment type for each
    fig = px.histogram(df_news,
                       x='source',
                       color='sentiment_type',
                       color_discrete_map={
                                             'neutral': 'silver',
                                             'positive': 'mediumspringgreen',
                                             'negative': 'orangered'
                                         }).update_xaxes(categoryorder="total descending")
    fig.update_layout(xaxis_title='',
                      font=dict(size=16),
                      width=810,
                      height=520,
                      legend=dict(orientation="h",
                                  yanchor="top",
                                  y=1.12,
                                  xanchor="center",
                                  x=0.5))
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 


with news_headline_word_cld:
    st.header('News Headlines WordCloud')
    
    df_news['news_headline_tokens'] = df_news['news_headline'].map(tokenizer)
    
    fig, ax = plt.subplots(figsize=(20, 10),
                           facecolor='w')  # facecolor='k' for black frame
    wordCloud = word_cloud(df_news['news_headline_tokens'].values)
    plt.imshow(wordCloud, interpolation='bilinear')
    ax.axis("off")
    fig.tight_layout(pad=0)
    st.pyplot(fig)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    
    
with kwrd_extract:
    st.header('Company Events Mentioned in the News')
    st.markdown("* Keyword extraction can be done in a number of ways, one of the most popular being RegEx. We will use a Python library called FlashText that is much quickier and easier to work with.")
    
    # get text for each news article
    df_news['news_text'] = df_news['url'].map(get_article_text)
    # cleaning news_text by transforming anything that is NOT space, letters, or numbers to ''
    df_news['news_text'] = df_news['news_text'].apply(lambda x: re.sub('[^ a-zA-Z0-9]', '', x))
    # check articles' text for our keywords 
    df_news['event_keywords'] = df_news['news_text'].map(keyword_extractor)
    
    # plot
    st.subheader('Number of News Articles Containg Company-Event Keywords')
    
    fig = px.histogram(
        df_news[df_news['event_keywords'] != ''],
        x='event_keywords',
        color='sentiment_type',
        color_discrete_map={
                             'neutral': 'silver',
                             'positive': 'mediumspringgreen',
                             'negative': 'orangered'
                         }).update_xaxes(categoryorder="total descending")

    fig.update_layout(yaxis_title='Count',
                      xaxis_title='',
                      width=810,
                      height=620,
                      font=dict(size=16),
                      showlegend=False)
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig)
    
    AgGrid(df_news[df_news['event_keywords'] != ''][['event_keywords', 'sentiment_type', 'source', 'news_headline', 'url']])
    

    