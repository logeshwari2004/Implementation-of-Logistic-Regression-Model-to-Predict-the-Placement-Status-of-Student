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

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student
Developed by:LOGESHWARI.P
RegisterNumber:212221230055

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
# Predicting for random value
clf.predict([[1	,78.33,	1,	2,	77.48,	2,	86.5,	0,	66.28]])
```

## Output:
## Dataset:
![271840286-46679c99-ac41-48eb-a089-37d536847389](https://github.com/logeshwari2004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94211349/85d452bc-dd81-41dd-a9d7-b5f4a6aee5ab)
## Dropping the unwanted columns:
![271841722-a39e88df-4cc6-4cff-b8ac-28b9006bf78d](https://github.com/logeshwari2004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94211349/2669245f-7f75-4128-ac71-b3efed0b8b8d)
## df.info():
![271840345-4641415b-00ec-4056-a04e-86686f0e7d41](https://github.com/logeshwari2004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94211349/4305d213-48f2-4ed8-8834-3b638150d915)
## df.info() after changing object into category:
![271840450-a663d998-c3bf-4d3a-950c-a06212808717](https://github.com/logeshwari2004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94211349/db4c5447-793b-477d-a21e-8c87ac0af76e)
## df.info() after changing into integer
![271840828-b044bd5a-7060-4817-b6ab-95df52f5fc47](https://github.com/logeshwari2004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94211349/99cedef3-7391-451c-b989-f5a23f7c967d)
## selecting features and lables
![271841166-4b7e1797-872e-4bad-be09-bb6456dee85a](https://github.com/logeshwari2004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94211349/a8be846d-e863-457b-b7e1-05f1993f9891)
## Training and testing
![271841325-298da98f-f1f6-4135-acfe-72a1dc1f148e](https://github.com/logeshwari2004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94211349/6eef2a65-745b-4560-ae2e-b92ae9426ee4)
## Creating a Classifier using Sklearn:
![271841494-563ba84b-c13b-4cc2-9b3b-04bdfa9834ef](https://github.com/logeshwari2004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94211349/58a5c74e-5e0c-45f3-ba06-56ace692dbe2)
## Predicting for random value:
![271841560-648f82a5-0837-4069-8ead-70c0aaac1d6b](https://github.com/logeshwari2004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94211349/9f7992d5-cca1-4f1c-8f02-5d7a538d79d8)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
