# stock_price_prediction_with_financial_news_sentiment
 combining LSTM time series model with NLP tools

### Executive Summary
In this project, I focus on the prediction of stock prices using deep learning time series models. 
Particularly, I examined the effect of news sentiment on improving model accuracy with the help of NLP tools. 
 
After examining stock prices of Apple (AAPL) and Microsoft (MSFT) in 2021 and related financial news, I found that FinBERT demonstrates the highest accuracy (69.51%) in classifying financial news sentiment. Moreover, model performance for AAPL improves after adding sentiment score as a feature, with maximum 10.4% increase in R-squared and maximum 39.8% decrease in MAE.
 
### Data and Libraries
  
Benchmark data used in this project for news sentiment analysis is converted from [Financial Phrasebank](https://huggingface.co/datasets/financial_phrasebank) by Huggingface. 
 
Stock prices of AAPL and MSFT are collected through [TTINGO API](https://api.tiingo.com/) for the duration of year 2021. 
 
Financial news headlines related to AAPL and MSFT are collected through [MARKETAUX API](https://www.marketaux.com/). 

Data analysis is implemented with Python 3.9.5 in framework `PyTorch` (version 1.11.0+cu113) and `Tensorflow` (ver. 2.8.2). Pre-trained models `FinBERT` from HuggingFace model hub is activated. Other libraries for model comparison include `Keras` from Tensorflow and `Vader` from NLTK (ver. 3.6.6). Libraries for data analysis and visualization include `sklearn` (ver.1.0.2), `pandas` (ver. 1.3.5) and `numpy` (ver. 1.12.6).   
  
### Hypotheses and background

__Hypothesis 1__: Compared to lexicon-based method [VADER](https://github.com/knuppe/vader), deep learning model [Keras](https://keras.io/api/) and fine-grained models [FinBERT](https://huggingface.co/yiyanghkust/finbert-tone) have higher accuracy in classifying financial news sentiment.
  
__Hypothesis 2__: Financial news sentiment is significantly correlated with daily closing prices of two chosen stock indices.

<p align="center">
  <img src="https://i.ibb.co/RbBZ6Xd/H2.png">
</p>





### Result Summary
Below is a table for F1 Scores of different models and feature sets.  
<p align="center">
  <img src="https://i.ibb.co/2sx43RR/image.png">
</p>