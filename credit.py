import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 

from sklearn.metrics import classification_report, accuracy_score  
from sklearn.metrics import confusion_matrix 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns

df=pd.read_csv('creditcard.csv')
'''
The most important columns are:
-- Time,
-- Amount,
-- and Class (fraud or not fraud).

data_df[‘Class’] = 0 Not a fraud transaction
data_df[‘Class’] = 1 Fraud transaction


#print(df.head())
#print(df.shape)
#print(df[['Amount','Time','Class']].describe())
#print(df.describe())  # statistical data
#print(df.columns)   # all the coluimns
#print(df.isna().any())  # gives true /false
'''
#print(df[df.columns[df.isna().sum()>0]].isna().sum().sort_values().plot.bar())
print(df[df.columns[df.isna().sum()]].isna().sum().sort_values().plot.bar())
plt.show()
'''
# to find null values and their percentage

null_columns=pd.DataFrame({'Columns':df.isna().sum().index,
                           'No. Null values':df.isna().sum().values,
                         'Percentage':df.isna().sum().values})#/df.shape[0]})
                           
#print(null_columns)

#print(df.isnull().sum())

#Percentage of total not fraud transaction

nfcount=0
notFraud=df['Class']
for i in range(len(notFraud)):
  if notFraud[i]==0:
    nfcount=nfcount+1

nfcount    
per_nf=(nfcount/len(notFraud))*100
#print('percentage of total not fraud transaction in the dataset: ',per_nf)

#Percentage of total fraud transaction

fcount=0
Fraud=df['Class']
for i in range(len(Fraud)):
  if Fraud[i]==1:
    fcount=fcount+1

fcount    
per_f=(fcount/len(Fraud))*100
#print('percentage of total fraud transaction in the dataset: ',per_f)


df=pd.DataFrame()
df['Fraud Transaction']=Fraud
df['Genuine Transaction']=notFraud
#print(df)

plt.title("Bar plot for Fraud VS Genuine transactions")
sns.barplot(x = 'Fraud Transaction', y = 'Genuine Transaction', data = df,
            palette = 'Blues', edgecolor = 'w')

#plt.show()

#As per the graph we can say the ratio of genuine transactions are higher than fraud transactions.

x=df['Amount']
y=df['Time']
plt.plot(x,y) 
plt.title('Time Vs Amount') 
sns.barplot(x = x, y = y, data = df, palette = 'Blues', edgecolor = 'w')
plt.show()

#the number high amount transactions are very low. So there is a high probability for huge transactions to be fraudulent .

plt.figure(figsize=(10,8), )
plt.title('Amount Distribution')
sns.distplot(df['Amount'],color='red');
plt.show()


# Correlation matrix 
correlation_metrics = df.corr() 
fig = plt.figure(figsize = (14, 9)) 
sns.heatmap(correlation_metrics, vmax = .9, square = True) 
plt.show() 


x=df.iloc[:,:-1]
y=df.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42) 

lr=LogisticRegression()

#fitting the model
lr.fit(x_train,y_train)

#predict
y_pred=lr.predict(x_test)
print("predicted values:",y_pred)

print("score is:",lr.score(x_test,y_test))

cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

#accuracy of the model
print("accuracy:",metrics.accuracy_score(y_test,y_pred)*100)
'''


