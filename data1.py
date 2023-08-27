import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
data1=pd.read_excel("lab.xlsx", sheet_name="Purchase data")
data1=data1.iloc[0:10,0:5]
data3=data1.iloc[0:10,1:5]
A=np.array(data1.iloc[0:10,1:4])
C=np.array(data1.iloc[0:10,4:5])
Ainv=np.linalg.pinv(A)
X=np.matmul(Ainv,C)
rank_data = np.linalg.matrix_rank(np.array(data3))
rankA=np.linalg.matrix_rank(A)
temp=np.array(data1['Payment (Rs)'])
n=len(temp)
print(temp)
Category=[]
count=0
for i in range(0,n):
    count=count+1
    if temp[i]>200:
        Category.append('Rich')
    else:
        Category.append('Poor')
data1.insert(loc = 5,column = 'Label',value = Category)
print(data1)
X = data1.drop(['Customer', 'Payment (Rs)', 'Label'], axis=1)
y = data1['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
print("Dimentionality of the Vector space : ",rank_data)
print("Rank of matrix A : ",rankA)
print("Cost of each procuct : ")
print(X)