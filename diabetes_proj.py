from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('diabetes.csv')
#print(df)

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=101)

model=LogisticRegression()

model.fit(x_train,y_train)

acc=model.score(x_test,y_test)
print(f"Accuracy={acc}")

#saving model
import joblib
joblib.dump(model,'diabetes_model.pkl')
