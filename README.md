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


# RESULT
        <<INCLUDE YOUR RESULT HERE>>
