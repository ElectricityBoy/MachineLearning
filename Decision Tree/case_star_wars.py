#%%

import pandas as pd
df = pd.read_parquet('data/dados_clones.parquet')

df.head()

#%%
df.columns.tolist()

df.groupby('General Jedi encarregado')['Status '].value_counts().unstack(fill_value=0)

#%%

df.head()


y = df['Status ']

df.columns.to_list()
features = [
 'Massa(em kilos)',
 'Estatura(cm)',
 'Distância Ombro a ombro',
 'Tamanho do crânio',
 'Tamanho dos pés',
 'Tempo de existência(em meses)',]


#%%

df['Distância Ombro a ombro'].unique()

df = df.replace({
    'Tipo 1': 1, 'Tipo 2':2, 'Tipo 3':3, 'Tipo 4':4, 'Tipo 5':5
})

df.head()

#%%
# Criando modelo

from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth = 3)
x = df[features]

model.fit(x,y)

#%%
import matplotlib.pyplot as plt

tree.plot_tree(model,feature_names=features, class_names= model.classes_, filled= True)

