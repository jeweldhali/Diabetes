import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


df = pd.read_csv("diabetes2.csv")


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
mod = LogisticRegression()

#x=df.iloc[:,0:8]
#y = df.iloc[:,8]

X = df.iloc[:,:-1].values
y = df.iloc[:,8:].values


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)



mod.fit(X_train,y_train)


pickle.dump(mod, open('model1.pkl','wb'))
