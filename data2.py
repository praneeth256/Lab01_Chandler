import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data2=pd.read_excel("lab.xlsx", sheet_name="IRCTC Stock Price")
columnD=data2.iloc[:,3:4]
meanD=np.mean(np.array(columnD))
varianceD=np.var(np.array(columnD))
data_wed=data2.loc[data2['Day']=='Wed']
data_wed_price=data_wed.iloc[:,3:4]
mean_wed=np.mean(np.array(data_wed_price))
data_apr=data2.loc[data2['Month']=='Apr']
data_apr_price=data_apr.iloc[:,3:4]
mean_apr=np.mean(np.array(data_apr_price))
data_chg=data2.iloc[:,8:9]
data_chg_array=np.array(data_chg)
data_chg_wed=np.array(data_wed.iloc[:,8:9])
n=len(data_chg_array)
neg_count=0
for i in range(0,n):
    if data_chg_array[i]<0:
        neg_count=neg_count+1
loss_prob=neg_count/n
wed_count=0
n1=len(data_chg_wed)
for i in range(0,n1):
    if data_chg_wed[i]<0:
        wed_count=wed_count+1
profit_wed=wed_count/n1
weddata = data2[data2['Day'] == 'Wed']
wedprofit = np.mean(weddata['Chg%'] > 0)
wedprob = np.mean(data2['Day'] == 'Wed')
cdprob = (wedprofit / wedprob)
print('Mean :', meanD)
print("Variance : ",varianceD)
print('Mean on wednesday :',mean_wed )
print('Mean in april:', mean_apr)
print("Probability of loss : ",loss_prob)
print("Probability of profit on wednesday : ",profit_wed)
print('Conditional Probability on Wednesday is:', cdprob)
plt.scatter(data2['Day'], data2['Chg%'])
plt.xlabel('Day')
plt.ylabel('Chg%')
plt.title('Chg% vs Day')
plt.show()
