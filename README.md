# Churn-Modeling-in-Python
Modeling churn probabilities for a data set of phone companies 20,000 customers. The model is a binary classifier giving 1 for yes they will churn (leave) and 0 for no they will not churn. The model uses mulitple methods to try and predict the likely outcome of whether a customer will churn from the phone company. First to train the models, the training data set was put through a 25% test/train split.
## Decision Tree by PCA
The first model used was a decision tree. In order to train this model, a PCA (principal component analysis) was done on the training set. This allows for dimensionallity reduction, and only takes into account the features that have the highest variance, therefore giving the most valuable information about the customer. The tree was set to a maximum depth of 4 to avoid overfitting. The code then also uses the classifier trained for the decision tree to directly use predict_prob to predict the associations as another numerical way of classifiying.
## Random Forest
The second model used was the random forest method. The maximum depth of 4 to avoid overfitting with the number of estimaters set to 1000. The code then also uses the classifier trained for the forest to directly use predict_prob to predict the associations as another numerical way of classifiying. 
## Logistic Regression
The last method used was logistic regression. This method required no other restraints as the previous two methods did, making it rather simple to apply to the train/test split. The code then also uses the classifier trained for the forest to directly use predict_prob to predict the associations as another numerical way of classifiying. 
## Averages
After the three models were trained and used to predict the outcomes, the predicted probabilities of the three models were then averaged together to see how well the associated probabilities were across the models. This as ended up actually resulting in the highest performing prediction.
## Performance Metrics
For all three of the models, the result of the training on the train/test split was visualized with an ROC curve and the resulting AUC score was found. In addition, the classification report using the '.score' was produced for each of the models.
