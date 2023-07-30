#importing libraries
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
#reading data from the file
df = pd.read_csv("data1.csv")
df=df.dropna(axis=1)
df['diagnosis'].value_counts()
#sns.countplot(x=df['diagnosis'],label="count")
# Label encoding(convert the value of M and B into 1 and 0)
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1]=labelencoder_Y.fit_transform(df.iloc[:,1].values)
#sns.pairplot(df.iloc[:,1:10],hue="diagnosis")
# Get the corelation
df.iloc[:,1:32].corr()
# Visualize the correlation
#plt.figure(figsize=(10,10))
#sns.heatmap(df.iloc[:,1:10].corr(),annot=True,fmt=".0%")
# Split the dataset into dependent and independent datasets
X=df.iloc[:,2:32].values
Y=df.iloc[:,1].values
Y=Y.astype('int')

# Spliting the data into training and test dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)

#print(X_test)
# Spl
# Feature scaling
from sklearn.preprocessing import StandardScaler
X_train=StandardScaler().fit_transform(X_train)
X_test=StandardScaler().fit_transform(X_test)
print(X_test)
# Algorithms/Models
def model(X_train,Y_train):
    # Logistic regression
    from sklearn.linear_model import LogisticRegression
    log=LogisticRegression(random_state=0)
    log.fit(X_train,Y_train)
    
    # Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    tree=DecisionTreeClassifier(random_state=0,criterion="entropy")
    tree.fit(X_train,Y_train)
    
    # Random Forest
    from sklearn.ensemble import RandomForestClassifier
    forest=RandomForestClassifier(random_state=0,criterion="entropy",n_estimators=10)
    forest.fit(X_train,Y_train)
    
    print('[0]logistic regression accuracy:',log.score(X_train,Y_train))
    print('[1]Decision tree accuracy:',tree.score(X_train,Y_train))
    print('[2]Random forest accuracy:',forest.score(X_train,Y_train))
    
    
    return log,tree,forest
model=model(X_train,Y_train)
# Testing the models/result

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

for i in range(len(model)):
    print("Model",i)
    print(classification_report(Y_test,model[i].predict(X_test)))
    print('Accuracy:',accuracy_score(Y_test,model[i].predict(X_test)))

# Prediction of random-forest
pred=model[2].predict(X_test)
print('Predicted values:')
print(pred)
print('Actual values:')
print(Y_test)

model_pkl_file = "breastcancer_model_RF_2.pkl" 
with open(model_pkl_file, 'wb') as file:  
    pickle.dump(model[2], file)
 


    