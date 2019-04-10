#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 14:37:07 2019

@author: alexpollack, mattmanganel
"""
#FILE USED FOR SUBMISSION

import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import csv
#import the training file as a pandas data frame
df = pd.read_csv('train.csv')
#import the testing file as a pandas data frame
test_df = pd.read_csv('test.csv')
#display the head of the training data frame
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#split the training data 1/4 test to 3/4 training
train, test = train_test_split(df, test_size = 0.25)
#set the training y as the churn column 'LEAVE' from the trainsplit of train df
train_y = train['LEAVE']
#set the testing y as the churn column 'LEAVE' from the testsplit of train df
test_y = test['LEAVE']
#set the testing x as the remaining df from the trainsplit of train df
train_x = train
#remove the column containing the chrun information
train_x.pop('LEAVE')
#set the testing y as the churn column 'LEAVE' from the testsplit of train df
test_x = test
#remove the column containing the chrun information
test_x.pop('LEAVE')

from sklearn.feature_extraction import DictVectorizer
#use the DictVectorizer to give numeric rank to the string info in the df's
vec = DictVectorizer()
vec.fit(df.to_dict('records'))
train_x = vec.transform(train_x.to_dict('records')).toarray()
test_x = vec.transform(test_x.to_dict('records')).toarray()
test_df = vec.transform(test_df.to_dict('records')).toarray()

##############################
#### DECISION TREE METHOD ####
##############################
#### this section used the decision tree model to predict chrun #####
print('\nDecision Tree')
#reduce dimensions by PCA before input to decision tree
pca = PCA(n_components = 5)
X_proj = pca.fit_transform(train_x) #training split x
T_proj = pca.transform(test_x)  #testing from split x
df_test_proj = pca.transform(test_df)   #test file data for prediction x
#set the decision tree classifier depth
treeclf = DecisionTreeClassifier(max_depth = 4) 
#train the classifier on the projected reduced data
treeclf.fit(X_proj, train_y)
#predict results with the training data 
r = treeclf.predict(T_proj)
#predict real results with the testing data file
r_df_test = treeclf.predict(df_test_proj)
#display head of predictions from training
print('r test set:',r_df_test[0:100])
#output results of the prediction on the true testing file for submission
out_df = pd.DataFrame(r_df_test, columns=['LEAVE'])
out_df['ID']= out_df.index
out_df.to_csv('test_results.csv',index=False)
#create ROC curve and find the AUC for the decision tree method's predictions
print('Decision Tree ROC')
fpr, tpr, thresholds = sk.metrics.roc_curve(test_y,treeclf.predict_proba(T_proj)[:,1])
plt.plot(fpr,tpr)
plt.show()
print('AUC: ',sk.metrics.auc(fpr,tpr))
print('Classification report')
print(classification_report(test_y,r))
print('Accuracy: ',treeclf.score(T_proj,test_y))

#this found probabilities of churn from decision tree predictions
prob_decisionTree_df_test = treeclf.predict_proba(df_test_proj)[:,1]
#output the results of the testing files prob. prediction for submisions
out_prob_decisionTree_df = pd.DataFrame(prob_decisionTree_df_test, columns=['LEAVE'])
out_prob_decisionTree_df['ID']= out_prob_decisionTree_df.index
out_prob_decisionTree_df.to_csv('proba_decisionTree_test_results.csv',index=False)

##############################
#### RANDOM FOREST METHOD ####
##############################
#### this section used the random forest model to predict chrun #####
print('\nRandom Forest')
#set the random forest classifier, number of iterants and max depth
rTreeClf = RandomForestClassifier(n_estimators=1000, max_depth = 4)
#train the classifier on the split data
rTreeClf.fit(train_x, train_y)
#predict outcome from training data
r_randTree = rTreeClf.predict(test_x)
#predict outcomes from true testing data file
randTree_df_test = rTreeClf.predict(test_df)

#output the results of the testing files prediction for submisions
out_randTree_df = pd.DataFrame(randTree_df_test, columns=['LEAVE'])
out_randTree_df['ID']= out_randTree_df.index
out_randTree_df.to_csv('randTree_test_results.csv',index=False)

#create ROC curve and find the AUC for the random forest method's predictions
print('Random Forest ROC')
fpr, tpr, thresholds = sk.metrics.roc_curve(test_y,rTreeClf.predict_proba(test_x)[:,1])
plt.plot(fpr,tpr)
plt.show()
print('AUC: ',sk.metrics.auc(fpr,tpr))
print('Classification report')
print('Accuracy: ',rTreeClf.score(test_x,test_y))

#this found probabilities of churn from random forest predictions
prob_randTree_df_test = rTreeClf.predict_proba(test_df)[:,1]
#output the results of the testing files prob. prediction for submisions
out_prob_randTree_df = pd.DataFrame(prob_randTree_df_test, columns=['LEAVE'])
out_prob_randTree_df['ID']= out_prob_randTree_df.index
out_prob_randTree_df.to_csv('proba_randTree_test_results.csv',index=False)


####################################
#### LOGISTIC REGRESSION METHOD ####
####################################
#### this section used the logistic regression model to predict chrun #####
print('\nLogistic Regression')
from sklearn.linear_model import LogisticRegression

#create the logistic regr classifier
logisticRegr = LogisticRegression()
#train the classifier on the training split data
logisticRegr.fit(X=train_x,y = train_y)
#test the prediction on the test splitn of the training data
test_y_pred = logisticRegr.predict(test_x)
#predict outcome from the true testing file using the classifier
real_pred = logisticRegr.predict(test_df)

#output the results of the testing files prediction for submisions
out_log_df = pd.DataFrame(real_pred, columns=['LEAVE'])
out_log_df['ID']= out_log_df.index
out_df.to_csv('log_test_results.csv',index=False)

#print testing results from the train/test split trial, AUC, ROC, f1, etc.
print('intercept: ' + str(logisticRegr.intercept_))
print('Regression: ' + str(logisticRegr.coef_))
print('Accuracy of log regression class. on test: {:.2f}'.format(logisticRegr.score(test_x,test_y)))
print('Classification report')
print(classification_report(test_y,test_y_pred))
print(df['LEAVE'].value_counts())
print('Log Regr ROC')
fpr, tpr, thresholds = sk.metrics.roc_curve(test_y,logisticRegr.predict_proba(test_x)[:,1])
plt.plot(fpr,tpr)
plt.show()
print('AUC: ',sk.metrics.auc(fpr,tpr))

#this found probabilities of churn from log regression predictions
prob_log_df_test = logisticRegr.predict_proba(test_df)[:,1]
#output the results of the testing files prob. prediction for submisions
out_prob_log_df = pd.DataFrame(prob_log_df_test, columns=['LEAVE'])
out_prob_log_df['ID']= out_prob_decisionTree_df.index
out_prob_log_df.to_csv('proba_logRegr_test_results.csv',index=False)

########################################################
#### AVERAGE OF THE THREE METHOD'S PROBABILITIES #######
########################################################
#### this section uses average of the prob. of each model's prediction #####
av_proba = prob_decisionTree_df_test + prob_randTree_df_test +prob_log_df_test
av_proba = av_proba/3
#output the results of the testing files av. prob. prediction for submisions
out_av_prob_df = pd.DataFrame(av_proba, columns=['LEAVE'])
out_av_prob_df['ID']= out_av_prob_df.index
out_av_prob_df.to_csv('average_proba_test_results.csv',index=False)








