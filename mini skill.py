#importing libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
#uploading dataset
df=pd.read_csv("C://Users//ayushi yadav//Downloads//cardio_train.csv//cardio_train.csv",sep=";")
print("shape of data",df.shape) 
print(df.head())
print("value count",df["cardio"].value_counts())
print(df.isnull().values.any())
print(df.isna().sum())
#cardio distribution graph
sns.countplot(data=df, x="cardio",hue="cardio")
plt.xlabel('cardio')
plt.ylabel('Count')
plt.title('Cardio Distribution')
plt.show()
#Gender Distribution Graph
sns.countplot(data=df, x='gender', hue="cardio",palette="colorblind",edgecolor="black")
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Distribution')
plt.show()
# Age Distribution
sns.countplot(data=df, x='age', hue="cardio",palette="colorblind",edgecolor="black")
plt.xlabel('age')
plt.ylabel('Count')
plt.title('Age Distribution')
plt.show()

df['yr']=(df['age']//365)
print(df['yr'])
#YearDistribution Graph
sns.countplot(data=df, x='yr', hue="cardio",palette="colorblind",edgecolor="black")
plt.xlabel('year')
plt.ylabel('Count')
plt.title('Year Distribution')
plt.show()
#correlation
corr_matrix=df.corr()
print(corr_matrix)
sns.heatmap(corr_matrix, annot=True)
plt.show()
df=df.drop(['yr'],axis=1)
#setting x and y variable
x=df.iloc[:,:-1]
print(x)
y=df.iloc[:,12]
print(y)
#splitting dataset
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)
#randomforest classifier
rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=5)
rf.fit(xtrain, ytrain)
rf_pred=rf.predict(xtest)
print("\npredicted value\n",rf_pred)
print("\naccuracy\n",rf.score(xtest,ytest)*100)
#confusion matrix
cm=confusion_matrix(rf_pred,ytest)
sns.heatmap(cm/np.sum(cm),annot=True,fmt='.2%',cmap="Blues")
# plt.show()
print("\nconfusion matrix\n",cm)
print("\nClassification Report:\n", classification_report(ytest, rf_pred)) 
print("AYUSHI YADAV,0901AI211016")
 # Plot feature importance
importances = rf.feature_importances_ 
indices = np.argsort(importances)[::-1] 
plt.figure(figsize=(10, 6)) 
sns.barplot(x=importances[indices], y=x.columns[indices])
plt.title('Feature Importance') 
# plt.show() 
n_trees = [10, 25, 50, 75, 100, 150, 200, 250, 300]
accuracies = [] 
for n in n_trees: 
    rf = RandomForestClassifier(n_estimators=n, max_depth=5, random_state=42)     
    rf.fit(xtrain, ytrain)     
    rf_pred = rf.predict(xtest)     
    accuracy = accuracy_score(ytest, rf_pred)    
    accuracies.append(accuracy) 
# Plot accuracy graph
plt.figure(figsize=(10, 6)) 
sns.lineplot(x=n_trees, y=accuracies)
plt.title('Random Forest Accuracy vs Number of Trees') 
plt.xlabel('Number of Trees') 
plt.ylabel('Accuracy') 
# plt.show() 







