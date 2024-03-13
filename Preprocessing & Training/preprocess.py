import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Preprocess:
    X = None
    threshold = 3
    categorical_features = []
    numerical_features = []
    outliers=[]

    def __init__(self):
        self.scaler = StandardScaler()

    def cleaning(self):
        X_cate = self.X.drop(columns = self.numerical_features)
        X_nume = self.X.drop(columns = self.categorical_features)
    
        print(X_cate.columns)
        print(X_nume.columns)

        return X_cate,X_nume
    
    def detect_outliers(self,df):
        # compute z = (point - sample mean) / sample standard deviation
        # if abs(z-score) > 3  ==> outlier
        self.outliers=[]
        for column in df.columns:
            Z_Score = (df[column] - np.mean(df[column])) / np.std(df[column])
            #print(Z_Score)
            column_outliers = df.index[np.abs(Z_Score) > self.threshold].tolist()
            self.outliers.extend(column_outliers)
        
        return self.outliers

    def Separate_features(self,X):
        if X is None:
            return 'No DataFrame Passed'

        self.X = X

        for column in X.columns:
            if len(X[column].value_counts()) <= 8:
                self.categorical_features.append(column)
                continue
                #pass
            self.numerical_features.append(column)

        X_cate,X_nume = self.cleaning()

        return X_cate,X_nume
        #return categorical_features, numerical_features
    
    def Standardization(self,X_train,X_test):
        cols = X_train.columns

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        X_train = pd.DataFrame(X_train,columns = cols)
        X_test = pd.DataFrame(X_test,columns = cols)

        return X_train,X_test

    def Split(self,X,y,test_size):
        X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=43,stratify=y,test_size=test_size)

        return X_train,X_test,y_train,y_test
    
    def concat(self,X_cate,X_nume):
        X_cate.reset_index(drop=True,inplace=True)
        X_nume.reset_index(drop=True,inplace=True)

        X_train = pd.concat([X_cate,X_nume],axis=1)

        return X_train
    


