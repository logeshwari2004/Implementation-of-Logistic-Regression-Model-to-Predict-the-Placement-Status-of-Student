# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Load the dataset and check for null data values and duplicate data values in the dataframe.

3.Import label encoder from sklearn.preprocessing to encode the dataset.

4.Apply Logistic Regression on to the model.

5.Predict the y values.

6.Calculate the Accuracy,Confusion and Classsification report.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:Logeshwari.P
RegisterNumber:  212221230055
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("Placement_Data.csv")
df
df.head()
df.tail()
df=df.drop(['sl_no','gender','salary'],axis=1)
df=df.drop(['ssc_b','hsc_b'],axis=1)
df.shape
df.info()
df["degree_t"]=df["degree_t"].astype("category")
df["hsc_s"]=df["hsc_s"].astype("category")
df["workex"]=df["workex"].astype("category")
df["status"]=df["status"].astype("category")
df["specialisation"]=df["specialisation"].astype("category")
df["degree_t"]=df["degree_t"].cat.codes
df["hsc_s"]=df["hsc_s"].cat.codes
df["workex"]=df["workex"].cat.codes
df["status"]=df["status"].cat.codes
df["specialisation"]=df["specialisation"].cat.codes
x=df.iloc[: ,:-1].values
y=df.iloc[:,- 1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

df.head()
from sklearn.linear_model import LogisticRegression

#printing its accuracy
clf=LogisticRegression()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") 
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion 

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
# Predicting for random value
clf.predict([[1	,78.33,	1,	2,	77.48,	2,	86.5,	0,	66.28]])

```

## Output:
## Dataset

![image](https://github.com/Mythilidharman/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119104110/f6c9fa6e-648f-4aea-ab60-3aa0ff50d253)

## Dropping the unwanted columns

![image](https://github.com/Mythilidharman/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119104110/6c6da4eb-e493-4104-96ce-6e28603dd1f1)
## df.info()
![image](https://github.com/Mythilidharman/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119104110/78da1d09-3c34-4c6d-963d-d65295d8419c)
## df.info() after changing object into category

![image](https://github.com/Mythilidharman/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119104110/23c98a0b-0c5c-4bfa-98b2-1ccb7a8873ea)
## df.info() after changing into integer
![image](https://github.com/Mythilidharman/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119104110/64b926aa-afc8-4c2a-9bd9-01121fa6e7e0)
## Selecting features and labels
![image](https://github.com/Mythilidharman/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119104110/55cb8d41-64e8-436b-9eaf-4b96f6c7a7cd)
## Training and testing
![image](https://github.com/Mythilidharman/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119104110/252f0699-6313-4607-89c5-82187f157b38)
## Creating a Classifier using Sklearn
![image](https://github.com/Mythilidharman/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119104110/89177e27-b560-45b1-ad9b-7dbf9e9f1d26)
## Confusion matrix and Classification report
![image](https://github.com/Mythilidharman/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119104110/30f9ea15-384e-4fa5-90d0-94a45995f0f3)
## Predicting for random value
![image](https://github.com/Mythilidharman/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119104110/cd0600e0-52da-4ba8-a768-63f6394e7175)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
