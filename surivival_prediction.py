import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold



class tatanic():
    def __init__(self,train_data,test_data):
        self.train_data=train_data
        self.test_data=test_data
        
        
    def summury_data(self):
        print("-------------------- Summury of Data --------------------------")
        data_head=train_data.head()
        print(data_head)
        decription=train_data.describe()
        print(decription)
        print(train_data.shape)
        data_info=train_data.info()
        print(data_info)
        print(train_data.isnull().sum())
        
        print(test_data.isnull().sum())
        
    def visualization_data(self,with_respect):
        survived_rate=train_data[train_data['Survived']==1][with_respect].value_counts()
        print(survived_rate)
        dead_rate=train_data[train_data['Survived']==0][with_respect].value_counts()
        print(dead_rate)
        
        data_frame=pd.DataFrame([survived_rate,dead_rate])
        print(data_frame)

        data_frame.index=['Survived_rate','dead_rate']
        
        data_frame.plot(kind='bar')
        
    def cleaning_data_name(self,train_test_data):
        for dataset in train_test_data:
            dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, 
                 "Major": 3, "Mlle": 3,"Countess": 3,"Ms": 3, 
                 "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, 
                 "Mme": 3,"Capt": 3,"Sir": 3 }
        
        for dataset in train_test_data:
            dataset['Title'] = dataset['Title'].map(title_mapping)
            
        # delete unnecessary feature from dataset
        train_data.drop('Name', axis=1, inplace=True)
        test_data.drop('Name', axis=1, inplace=True)
                    
        
            
     
    def cleaning_data_sex(self,train_test_data):
        sex_mapping = {"male": 0, "female": 1}
        for dataset in train_test_data:
            dataset['Sex'] = dataset['Sex'].map(sex_mapping)
    
    def cleaning_data_age(self,train_test_data):
        train_data["Age"].fillna(train_data.groupby("Title")["Age"].transform("median"), inplace=True)
        test_data["Age"].fillna(test_data.groupby("Title")["Age"].transform("median"), inplace=True)
        train_data.groupby("Title")["Age"].transform("median")
        for dataset in train_test_data:
            dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
            dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
            dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
            dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
            dataset.loc[ dataset['Age'] > 62, 'Age'] = 4
            
            
    def cleaning_data_embarked(self,train_test_data):
        for dataset in train_test_data:
            dataset['Embarked'] = dataset['Embarked'].fillna('S')
            
        embarked_mapping = {"S": 0, "C": 1, "Q": 2}
        for dataset in train_test_data:
            dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
            
    

       
    def cleaning_data_fare(self,train_test_data):
        # fill missing Fare with median fare for each Pclass
        train_data["Fare"].fillna(train_data.groupby("Pclass")["Fare"].transform("median"), inplace=True)
        test_data["Fare"].fillna(test_data.groupby("Pclass")["Fare"].transform("median"), inplace=True)
        for dataset in train_test_data:
            dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
            dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
            dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
            dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3
            
    def cleaning_data_cabin(self,train_test_data):
        for dataset in train_test_data:
            dataset['Cabin'] = dataset['Cabin'].str[:1]
            
        cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
        for dataset in train_test_data:
            dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
            
        # fill missing Fare with median fare for each Pclass
        train_data["Cabin"].fillna(train_data.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
        test_data["Cabin"].fillna(test_data.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
        train_data["FamilySize"]=np.nan
        
        
    
    def cleaning_data_family(self,train_test_data):
        train_data["FamilySize"]=train_data["SibSp"]+train_data["Parch"] +1
        test_data["FamilySize"]=test_data["SibSp"] + test_data["Parch"] + 1
        family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
        for dataset in train_test_data:
            dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)
        #family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
        #for dataset in train_test_data:
         #   dataset['FamilySize']=dataset['FamilySize'].map(family_mapping)
        
        #features_drop=['Ticket', 'SibSp', 'Parch']
        #train_data=train_data.drop(features_drop,axis=1)
        #test_data=test_data.drop(features_drop,axis=1)
        train_data.drop('Ticket', axis=1, inplace=True)
        test_data.drop('Ticket', axis=1, inplace=True)
        
        train_data.drop('SibSp', axis=1, inplace=True)
        test_data.drop('SibSp', axis=1, inplace=True)
        
        train_data.drop('Parch', axis=1, inplace=True)
        test_data.drop('Parch', axis=1, inplace=True)
        
        train_data.drop('PassengerId', axis=1, inplace=True)
        
        
    def Modelling(self,predict_data):
        k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
        clf = KNeighborsClassifier(n_neighbors = 21)
        target=train_data['Survived']
        train_data.drop('Survived', axis=1,inplace=True)
        score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring='accuracy')
        print(score)
        print(round(np.mean(score)*100, 2))
        print(target)
        
        clf.fit(train_data,target)
        
        passenger_id=test_data['PassengerId']
        test_data.drop('PassengerId',axis=1,inplace=True)
        prediction=clf.predict(test_data)
        print(prediction)
        print(passenger_id)
        predict_data['PassengerId']=passenger_id
        predict_data['Survived']=prediction
        predict_data.to_csv('file1.csv') 

        
        
    
        
     



            
train_data=pd.read_csv("C:/Users/Nishant Nimbhorkar/Desktop/Data science data/predective analysis/kaggle comp/tatanic/train.csv")
test_data=pd.read_csv("C:/Users/Nishant Nimbhorkar/Desktop/Data science data/predective analysis/kaggle comp/tatanic/test.csv")
t=tatanic(train_data,test_data)
train_test_data=[train_data,test_data]
predict_data=pd.DataFrame(
                          {'PassengerId':[],
                           'Survived':[]
                           }
                          )
t.summury_data()
t.cleaning_data_name(train_test_data)
t.cleaning_data_sex(train_test_data)
t.cleaning_data_age(train_test_data)
t.cleaning_data_embarked(train_test_data)
t.cleaning_data_fare(train_test_data)
t.cleaning_data_cabin(train_test_data)
t.cleaning_data_family(train_test_data)
t.summury_data()
t.visualization_data('Sex')
t.Modelling(predict_data)



    






