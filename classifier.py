from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
import re


class ensemble():
    def __init__(self, models: list):
        self.models = models
        self.blender = LogisticRegression()  ## инициализация блендера 
    def fit(self, X, y):
        meta_X = list()         

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
        for model in self.models:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            pred = pred.reshape(len(pred), 1)
            meta_X.append(pred)
        meta_X = np.hstack(meta_X)
        self.blender.fit(X_test, y_test)
        
    
    def predict(self, X):
        pred = self.blender.predict(X)
        return pred
    


class classifier():
    def __init__(self,models):
        self.models = models
        self.ensemble = ensemble(self.models)
        
    


    

    