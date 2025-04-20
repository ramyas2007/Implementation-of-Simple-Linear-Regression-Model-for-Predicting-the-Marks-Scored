# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: RAMYA S
RegisterNumber:  212224040268
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("/content/student_scores (1).csv")
df.head()
```
```
df.tail()
```
```
x=df.iloc[:,:-1].values
x
```
```
y=df.iloc[:,1].values
y
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
```
```
y_test
```
```
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train, regressor.predict(x_train), color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
plt.scatter(x_test,y_test,color="purple")
plt.plot(x_test, regressor.predict(x_test), color="yellow")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ", rmse)
```

## Output:

![image](https://github.com/user-attachments/assets/ab80b68a-03a7-4f25-a914-101dd0308bb5)

![image](https://github.com/user-attachments/assets/07312d43-d156-486e-92f2-1deec164e4a3)

![image](https://github.com/user-attachments/assets/167bcb82-211c-4b70-990d-b40a568588be)

![image](https://github.com/user-attachments/assets/a12db859-6890-4b94-a1b1-a643150bf7a8)

![image](https://github.com/user-attachments/assets/e1bd91a1-55c6-4c0c-8b2f-c814971103f8)

![image](https://github.com/user-attachments/assets/22a92284-c7eb-4645-9c1f-5a80b204432a)

![image](https://github.com/user-attachments/assets/faa34d2d-f51a-4a2e-9529-ba4b339fc46c)

![image](https://github.com/user-attachments/assets/361687dd-50de-4d1b-a2ab-e324c7fafbe6)

![image](https://github.com/user-attachments/assets/076ed27a-4cc0-4d9d-be09-0f1b7369fda4)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
