# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 18:58:23 2020

@author: YASHESWI MISHRA
"""



import pandas as pd 
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
bankdata=pd.read_csv("bankyes.csv",sep=";")

##Checking null values
bankdata.isnull().sum()

#Remove columns that do not act as good predictors for output variable
bankdata.drop(["day","contact","duration"],axis=1,inplace=True)

#Convert two class categorical data into numeric data
bankdata["default"]=bankdata["default"].apply(lambda x:1 if x=="yes" else 0)
bankdata["housing"]=bankdata["housing"].apply(lambda x:1 if x=="yes" else 0)
bankdata["loan"]=bankdata["loan"].apply(lambda x:1 if x=="yes" else 0)
bankdata["y"]=bankdata["y"].apply(lambda x:1 if x=="yes" else 0)

#Convert categorical data with more than 2 classes into dummy variables
bankdata=pd.get_dummies(bankdata,columns=["job","marital","education","month","poutcome"])
imp=bankdata["y"]
bankdata.drop(["y"],axis=1,inplace=True)
bankdata=pd.concat([bankdata,imp],axis=1)
##Splitting data into train and test
from sklearn.model_selection import train_test_split
##train,test = train_test_split(bankdata,test_size=0.3)
X = bankdata.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]]
Y = bankdata.iloc[:,43]

from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])

# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))
os_data_X.shape
X_train.shape
X_test.shape

x=os_data_X
y=os_data_y
import statsmodels.api as sm
logit_model=sm.Logit(y,x)
result=logit_model.fit()
print(result.summary())
##In the above code, we find that 2 variables have probability more than 0.05.Thus we will discard it
x.drop(["education_tertiary","default"],axis=1,inplace=True)
logit_model=sm.Logit(y,x)
result=logit_model.fit()
x.shape
##Building model on train dataset
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
classifier = LogisticRegression()
X_train.shape
y_train.shape
classifier.fit(X_train,y_train)
classifier.coef_
classifier.predict_proba (X_train)
y_pred = classifier.predict(X_train)
y_pred
y_train["y_predicted"]=y_pred
y_prob = pd.DataFrame(classifier.predict_proba(X_train.iloc[:,:]))

confusion_matrix =pd.crosstab(y_train["y"],y_train.y_predicted)
print (confusion_matrix)
accuracy = (17121+15393)/(17121+2483+4155+15393)
accuracy  #83.04
##ROC model for training data
from sklearn import metrics
from sklearn.metrics import classification_report
classification_report(y_train.y,y_train.y_predicted)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_train.y,y_train.y_predicted)
fpr, tpr, thresholds = roc_curve(y_train.y,y_train.y_predicted)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


#Test data
classifier1 = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
classifier1 = LogisticRegression()
X_train.shape
y_train.shape
classifier1.fit(X_test,y_test)
classifier1.coef_
classifier1.predict_proba (X_test)
y_pred1 = classifier1.predict(X_test)
y_pred1
y_test["y_predicted1"]=y_pred1
y_prob1 = pd.DataFrame(classifier1.predict_proba(X_test.iloc[:,:]))

confusion_matrix1 =pd.crosstab(y_test["y"],y_test.y_predicted1)
print (confusion_matrix1)
accuracy = (6998+6808)/(6998+6808+1364+1610)
accuracy  #82.27
##ROC model for testing data
from sklearn import metrics
from sklearn.metrics import classification_report
classification_report(y_test.y,y_test.y_predicted1)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test.y,y_test.y_predicted1)
fpr, tpr, thresholds = roc_curve(y_test.y,y_test.y_predicted1)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()




































