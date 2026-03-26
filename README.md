# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages.

2.Analyse the data.

3.Use modelselection and Countvectorizer to preditct the values.

4.Find the accuracy and display the result. 
 

## Program:

/*
Program to implement the SVM For Spam Mail Detection..
Developed by: VIJAYASHANKAR N
RegisterNumber: 212225230301
*/
```
import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```
## Output:
<img width="894" height="603" alt="image" src="https://github.com/user-attachments/assets/4fd0eaa3-deca-4a84-bd03-c2f5cdaff16f" />
<img width="873" height="146" alt="image" src="https://github.com/user-attachments/assets/5d2e6026-10c3-41d8-8bb7-06f8d2e8235a" />
<img width="878" height="157" alt="image" src="https://github.com/user-attachments/assets/cdc6d81b-b58d-49f6-a539-244312b75579" />
<img width="895" height="341" alt="image" src="https://github.com/user-attachments/assets/c7a1ed86-f6dd-467d-8107-f786a595488e" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
