# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
coding:
        from google.colab import drive
drive.mount('/content/drive')

ls drive/MyDrive/'Colab Notebooks'/DATA/

# **Exploratory data analysis**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
%matplotlib inline
cf.go_offline()

titan=pd.read_csv('drive/MyDrive/Data Science/titanic_dataset.csv')

titan.head()

titan.isnull()

sns.heatmap(titan.isnull(),yticklabels=False,cbar=False,cmap = 'viridis')

sns.set_style('whitegrid')
sns.countplot(x='Survived',data=titan,palette='RdBu_r')

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=titan,palette='RdBu_r')

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=titan,palette='rainbow')

sns.displot(titan['Age'].dropna(),kde=False,color='darkred',bins=40)

titan['Age'].hist(bins=30,alpha=0.3)

sns.countplot(x='SibSp',data=titan)

titan['Fare'].hist()

titan['Fare'].iplot()

plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=titan,palette='winter')

def impute_age(cols):
  Age=cols[0]
  Pclass=cols[1]
  if pd.isnull(Age):
    if Pclass == 1:
      return 37
    elif Pclass == 2:
      return 29
    else:
      return 24
  else:
    return Age

titan['Age'] = titan[['Age','Pclass']].apply(impute_age,axis=1)

sns.heatmap(titan.isnull(),yticklabels=False,cbar=False,cmap='viridis')

titan.drop('Cabin',axis=1,inplace=True)

titan.head()

titan.dropna(inplace=True)

titan.info()

pd.get_dummies(titan['Embarked'],drop_first=True).head()

sex=pd.get_dummies(titan['Sex'],drop_first=True)
embark=pd.get_dummies(titan['Embarked'],drop_first=True)

titan.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

titan.head()

titan=pd.concat([titan,sex,embark],axis=1)

titan.head()

titan.drop('Survived',axis=1).head()

titan['Survived'].head()

from sklearn.model_selection import train_test_split

X_titan,X_test,Y_titan,Y_test = train_test_split(titan.drop('Survived',axis=1),titan['Survived'],test_size=0.30,random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()
logmodel.fit(X_titan,Y_titan)

predictions = logmodel.predict(X_test)

from sklearn.metrics import confusion_matrix

accuracy=confusion_matrix(Y_test,predictions)

accuracy

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(Y_test,predictions)
accuracy

predictions

Output:
![image](https://github.com/user-attachments/assets/890a183e-62d1-4842-85e1-2b77427e2494)
![image](https://github.com/user-attachments/assets/45e68a67-3d8b-4bde-8c15-1dfaf91e08cf)
![image](https://github.com/user-attachments/assets/5840ffda-4589-4df6-a2eb-c9277e091d0b)
![image](https://github.com/user-attachments/assets/0b0063bc-7099-4b57-b617-5507f34c9d73)
![image](https://github.com/user-attachments/assets/90e3da02-e45e-4e3a-9a59-82f62036a391)
![image](https://github.com/user-attachments/assets/627e4dd1-4df7-49ea-bc00-16aefc86c958)
![image](https://github.com/user-attachments/assets/811bb791-0794-49ac-a444-fe47cb603073)
![image](https://github.com/user-attachments/assets/19a4266b-3fa1-412e-b2a8-4bfc63f2a69b)
![image](https://github.com/user-attachments/assets/bc651add-1af1-45da-af34-a4d62f867483)
![image](https://github.com/user-attachments/assets/8278c6ae-95e2-4d5a-ab64-2c6eeaf3f4d3)
![image](https://github.com/user-attachments/assets/40c44174-248e-40d2-accf-6d1b6619521e)
![image](https://github.com/user-attachments/assets/bd275956-b860-4388-844f-d8b194e3bd42)
![image](https://github.com/user-attachments/assets/551a9019-9580-409c-ba0c-b57a506491e1)
![image](https://github.com/user-attachments/assets/84c6d050-5f54-4bc4-9798-5ea2af315e1c)
![image](https://github.com/user-attachments/assets/dd676729-6c2c-461e-9d43-d1374a0fcf0b)
![image](https://github.com/user-attachments/assets/4ac51547-2df8-4a51-817a-595a67634fdc)
![image](https://github.com/user-attachments/assets/e57e1d52-fd1f-43f1-95c0-04b667dcc59a)
![image](https://github.com/user-attachments/assets/2bf66767-db5b-4581-b10a-7806c1c67d0d)
![image](https://github.com/user-attachments/assets/0f28c292-dc1d-4790-a3e3-84e52f720a96)
![image](https://github.com/user-attachments/assets/eb44617e-1798-43e4-88f6-41ed53e60baf)
![image](https://github.com/user-attachments/assets/7dab067a-90d5-4b88-8e8c-f11b7953f150)


# RESULT
        Data analysis was completed successfully
