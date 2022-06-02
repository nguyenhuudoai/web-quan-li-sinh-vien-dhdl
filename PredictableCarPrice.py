import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics 


df_home = pd.read_csv("car.csv", encoding='cp1252')
# print (df_home.shape)
# print (df_home.head(10))
list = ["year" , "condition" , "mmr"]
# cot x 3 du lieu tren
X = df_home[list]
print (X.head(500))
# cot y gia ban 
y = df_home["sellingprice"]
print (y.head(500))
# chuan hoa va in ra du lieu
from sklearn.decomposition import PCA 
X = PCA(1).fit_transform(X)
print (X[:500])

# chia tap du lieu train, test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=35)

from sklearn import linear_model
regr = linear_model.LinearRegression().fit(X_train, y_train)

y_pred = regr.predict(X_test)
# cac he so
from sklearn.metrics import mean_squared_error, r2_score
# The coefficients
print('Coefficients: \n', regr.coef_)
print('Bias: \n', regr.intercept_)
# The mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))


plt.scatter(y_test, y_pred)
plt.show()

# dung hoi quy tuyen tinh de du doan gia xe
plt.scatter(X_train, y_train, color='green')
plt.scatter(X_train, regr.predict(X_train), color='red')
plt.scatter(X_test[:10,:], y_test[:10], color='black')
plt.title('Linear regression for car Price')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
# so sanh y test va y du doan
plt.plot([min(y_test), max(y_test)],[min(y_pred),max(y_pred)])
plt.scatter(y_test, y_pred, color='red')
plt.title('Compare')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()

print("Nhập số năm sảm xuất :")
year=float(input())

print("Nhập điểm đánh giá :")
condition=float(input())

print("Nhập số km đã đi :")
mmr=float(input())

list1 = [ year , condition , mmr]

x = statistics.mean(list1)

need_prediction = [x]
for elem in need_prediction:
    print(regr.predict([[elem]]))
