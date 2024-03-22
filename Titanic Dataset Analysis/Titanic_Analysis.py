import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# data gathering
# import data
# analysis

# Checkout head data

train = pd.read_csv('titanic_train.csv')
print(train.head())

#Checkout for null values

print(train.isnull())

# data visualisation for missing data
# null data color map and color bar to remove color bar, yticklabels to add or remove y
# axis labels

#sns.heatmap(train.isnull(),yticklabels=False,cmap='viridis',cbar=False)

# grid styling

#sns.set_style('whitegrid')

# count plot is used to count

# sns.countplot(x='Survived',data=train)
# Not Survived - 549, Survived - 341
# hue - category by color , pallete - color combination
# gender wise data visualisation

# sns.countplot(x='Survived',data=train,hue='Sex',palette='RdBu_r')
# Passenger class data visualisation

# sns.countplot(x='Survived', data=train,hue='Pclass')

# distribution plot
#kde - it is the kernal density of the graph and bins reduce the size of the bars

# sns.displot(train['Age'].dropna(),kde=False,bins=30)

# sns.countplot(x='SibSp',data=train)
# train['Fare'].hist(bins=30,figsize=(10,4))

# to visualise mean we use boxplot

# sns.boxplot(x='Pclass',y='Age',data=train)

# Data Cleaning

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if(Pclass == 1):
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
       return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
train.drop('Cabin',axis=1,inplace=True)

sns.heatmap(train.isnull(),yticklabels=False,cmap='viridis',cbar=False)
plt.show()