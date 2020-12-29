import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import pickle


data = pd.read_csv(r'C:\Users\vamsi\Desktop\ml-deployment\train.csv')
data = data.dropna(axis=0)


train_input = np.array(data.iloc[0:500,0]).reshape(500,1)
train_output = np.array(data.iloc[0:500,1]).reshape(500,1)

test_input = np.array(data.iloc[500:700,0]).reshape(199,1)
test_output = np.array(data.iloc[500:700,1]).reshape(199,1)

from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression()
linear_regressor.fit(train_input,train_output)


pickle.dump(linear_regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2]]))

