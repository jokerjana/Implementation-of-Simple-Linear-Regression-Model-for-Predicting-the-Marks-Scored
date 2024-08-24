# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: JANARTHANAN B
RegisterNumber: 212223100014
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:
![image](https://github.com/user-attachments/assets/d5e9a552-6a47-4156-8185-89497356b603)

![image](https://github.com/user-attachments/assets/3510adf3-9266-4fde-a5ee-1f87d33508cd)

![image](https://github.com/user-attachments/assets/b65955c7-4f8e-4f20-986c-6e21d0ed4b43)

![image](https://github.com/user-attachments/assets/dddbafae-eed6-439b-9c78-caaa459e2f81)

![image](https://github.com/user-attachments/assets/0427bebc-b39b-4345-90dc-4daafbca3d72)

![image](https://github.com/user-attachments/assets/0e7c38a0-9dff-4a3f-9f4a-384b2170d6ab)

![image](https://github.com/user-attachments/assets/a0ff0bd6-f19b-4c7d-b500-316a47ebe2a0)

![image](https://github.com/user-attachments/assets/63bb195d-a859-4694-a6cd-bcecaefc53d9)

![image](https://github.com/user-attachments/assets/b05459c2-5061-4277-bb98-779d8ae1af96)

![image](https://github.com/user-attachments/assets/3eb88a00-d0b9-4edf-8ff6-2dc8c90db2c6)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
