# assignment3

#problem1
# importing libraries
import pandas as pd 
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt

df=pd.read_csv("train.csv") # reading train dataset

df.head() #getting the rows from dataset

le = preprocessing.LabelEncoder()  #finding the correlation between survived and sex 
df['Sex'] = le.fit_transform(df.Sex.values)
df['Survived'].corr(df['Sex'])

matrix = df.corr() # printing the correlation matrix
print(matrix)

df.corr().style.background_gradient(cmap="Blues") #colourmap for visualization 1

sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()  #heatmap for visualization 2

#NAive bais

train_raw = pd.read_csv('train.csv')
test_raw = pd.read_csv('test.csv')

# Join data to analyse and process the set as one.
train_raw['train'] = 1
test_raw['train'] = 0
df = train_raw.append(test_raw, sort=False)




features = ['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp']
target = 'Survived'

df = df[features + [target] + ['train']]
# Categorical values need to be transformed into numeric.
df['Sex'] = df['Sex'].replace(["female", "male"], [0, 1])
df['Embarked'] = df['Embarked'].replace(['S', 'C', 'Q'], [1, 2, 3])
train = df.query('train == 1')
test = df.query('train == 0')



# Drop missing values from the train set.
train.dropna(axis=0, inplace=True)
labels = train[target].values


train.drop(['train', target, 'Pclass'], axis=1, inplace=True)
test.drop(['train', target, 'Pclass'], axis=1, inplace=True)


from sklearn.model_selection import train_test_split, cross_validate #importing train_test_split(),cross_validate from the library 

X_train, X_val, Y_train, Y_val = train_test_split(train, labels, test_size=0.2, random_state=1)

import warnings  #importing libraries 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix

%matplotlib inline
# Suppress warnings
warnings.filterwarnings("ignore")

classifier = GaussianNB() # train the classifier with X_train, Y_train
 
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_val) #finding the accuracy

# Summary of the predictions made by the classifier
print(classification_report(Y_val, y_pred))
print(confusion_matrix(Y_val, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_val, y_pred))


#problem_2
glass=pd.read_csv("glass.csv") #importing glass dataset

glass.head() #getting the rows from dataset

glass.corr().style.background_gradient(cmap="Reds")#colourmap for visualization 1

sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag') #colourmap for visualization 2
plt.show()

features = ['Rl', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
target = 'Type'


X_train, X_val, Y_train, Y_val = train_test_split(glass[::-1], glass['Type'],test_size=0.2, random_state=1)

classifier = GaussianNB()

classifier.fit(X_train, Y_train) # train the classifier with X_train, Y_train


y_pred = classifier.predict(X_val)

# Summary of the predictions made by the classifier
print(classification_report(Y_val, y_pred)) #used to measure the quality of predictions from a classification algorithm
print(confusion_matrix(Y_val, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_val, y_pred)) #finding the accuracy

from sklearn.svm import SVC, LinearSVC #Implement linear SVM method using scikit library

classifier = LinearSVC()

classifier.fit(X_train, Y_train)
# train the classifier with X_train, Y_train

y_pred = classifier.predict(X_val)

# Summary of the predictions made by the classifier
print(classification_report(Y_val, y_pred)) #used to measure the quality of predictions from a classification algorithm
print(confusion_matrix(Y_val, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score 
print('accuracy is',accuracy_score(Y_val, y_pred))  #finding the accuracy
