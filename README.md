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

__Hypothesis 3__: A fine-tuned LSTM model integrating financial news sentiment has higher accuracy in predicting market trend than the one without.

<p align="center">
  <img src="https://i.ibb.co/YkPfWVL/H3.png">
</p>


### Experiment Design

The project is designed for three phases. 
In __Phase 1__, accuracy of news sentiment classification is tested on the benchmark dataset with VADER, Keras and FinBERT respectively. 
In __Phase 2__, Pearson correlation between news sentiment scores and financial indicators are plotted for the two stocks.  
In __Phase 3__, two models are implemented to predict stock prices, with one based on financial indicators only and the other including sentiment scores as a feature.  

<p align="center">
  <img src="https://i.ibb.co/HBbjfnC/EDesign.png">
</p>
 
<p align="center">
  <img src="https://i.ibb.co/CJcjqrr/Ed-2.png">
</p>

### Results

Results of two models are evaluated. The first model includes only 10 financial indicators as features to predict next day close price, using the data from 1 day and 10 days respectively.  Fig 13-1 and Fig 13-2 visually present the training result for stock AAPL. 
 
<p align="center">
  <img src="https://i.ibb.co/KqzPcRs/Fig13.png">
</p>

In the second model, I add one additional feature ‘sentiment score’ and examine whether it contributes to the model accuracy in predicting next day close price of the two stocks. Fig 15-1 and Fig 15-2 visually present the training result of the second model for stock AAPL. 

<p align="center">
  <img src="https://i.ibb.co/SnmtcBg/Fig15.png">
</p>

The result suggests that model performance for AAPL improves after adding sentiment score as a feature, with maximum 10.4% increase in R-squared and maximum 39.8% decrease in MAE. For MSFT, however, the results suggest an opposite direction. Although the first model achieves a relatively high R-squared (91.5% for 1-day model and 89.9% for 10-day model), the second model with sentiment score does not contribute to accuracy. 
