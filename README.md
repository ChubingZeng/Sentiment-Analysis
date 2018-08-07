# Sentiment Analysis for 400,000 Amazon Reviews

## Description 
In this project, the goal is to perform sentiment analysis to determine whether a review is positive or negative. 
I implemented 3 different machine learning algorithms to build text classifiers for Amazon reviews. The three algorithms are: neural networks (LSTM to be specific), decision tree and Naive Bayes. 

## Dataset 
The data I'm using comes from the [Kaggle Amazon review competition](https://www.kaggle.com/bittlingmayer/amazonreviews). 

## Analysis Result
The LSTM model performs the best (AUC 0.96) but took the longest to train.  
![roc_lstm](https://user-images.githubusercontent.com/23446412/43758052-bf97af50-99cf-11e8-85fc-83adb9d3f0d0.png)

Please refer to the *.py* files for my code, and *analysis report.pdf* for detailed description of how I pre-processed the data, built up the models and compared the performance of the three methods. 

In this [5-min video](https://drive.google.com/file/d/1ehwHWsjUm3UTG_I5N7zvbY4dlMfkIYn1/view?usp=sharing), I described in detail the dataset, preprocesssing, classifications, results and discussion of the problem. 
