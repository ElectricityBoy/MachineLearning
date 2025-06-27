#%%
import kagglehub

# Download latest version
path = kagglehub.dataset_download("esathyaprakash/electrical-fault-detection-and-classification")

print("Path to dataset files:", path)

#%%
import pandas as pd

data = pd.read_csv('data/classData.csv')

data.head()

data.info()

#%%

data.columns.to_list()

target = ['G', 'C', 'B', 'A']
features = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']


#%%

from sklearn import tree

model = tree.DecisionTreeClassifier(max_depth=5)

x = data[features]
y = data[target]

model.fit(x,y)

#%%

import matplotlib.pyplot as plt

plt.figure(dpi = 700, figsize=[6,6])

tree.plot_tree(model, feature_names= features, class_names= model.n_classes_, filled= True)

#%%

model.predict([[-643.663617	,-224.159427,-132.282815,0.209537,-0.095554, -0.113983]])