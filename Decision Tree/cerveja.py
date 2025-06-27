#%%

import pandas as pd

df = pd.read_excel('data/dados_cerveja.xlsx')

df.head()

#%%

from sklearn import tree

y = df['classe']

df.columns.tolist()

caracteristicas = ['temperatura', 'copo', 'espuma', 'cor']

x = df[caracteristicas]

#%%

x = x.replace({
    "mud":1, "pint":2,
    "sim":1, "não":0,
    "escura":1,"clara":0
})

#%%

model = tree.DecisionTreeClassifier()

model.fit(x,y)


#%%
import matplotlib.pyplot as plt

tree.plot_tree(model,feature_names=caracteristicas,class_names=model.classes_,filled= True)

