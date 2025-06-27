#%%

import kagglehub

# Download latest version
path = kagglehub.dataset_download("himanshunakrani/student-study-hours")

print("Path to dataset files:", path)

#%%

import pandas as pd

df = pd.read_csv(r'C:\Users\elder\Desktop\Cursos\Machine Learning\MachineLearning\data\score.csv')

df.head()

df.info()

df.describe()

#%%
#Definindo metricas de analise

df.columns.to_list()

y = df['Scores']

X = df[['Hours']]

#%%

from sklearn import linear_model
from sklearn import tree
import numpy as np

X_sorted = np.linspace(X.min(), X.max(), 100)

model = linear_model.LinearRegression(fit_intercept=True)
model.fit(X,y)
predict = model.predict(X_sorted)

model2 = tree.DecisionTreeRegressor(random_state= 42, max_depth=2)
model2.fit(X,y)
tree_predict = model2.predict(X_sorted)
#%%
a,b = model.intercept_, model.coef_[0]
print(a,b)
#%%

import matplotlib.pyplot as plt

plt.plot(X['Hours'],y, 'o')
plt.plot(X_sorted,predict)
plt.plot(X_sorted,tree_predict)
plt.grid(True)
plt.title('Cerveja vs Nota')
plt.xlabel('Qtd cervejas')
plt.ylabel('Nota')
plt.legend(['Dados colhidos',f'{a:.3f} + {b:.3f}x','Tree'])

