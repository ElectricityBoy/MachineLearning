#%%
import pandas as pd

df = pd.read_excel(r'C:\Users\elder\Desktop\Cursos\Machine Learning\MachineLearning\data\dados_cerveja_nota.xlsx')

df.head()

#%%

X = df[['cerveja']] # X é sempre uma matriz


y= df['nota'] # É um vetor - series

#%% 

from sklearn import linear_model

reg = linear_model.LinearRegression(fit_intercept=True)

reg.fit(X,y)
#%%
a,b = reg.intercept_, reg.coef_[0]

print(a,b) # Melhor ajuste

#%%

predict = reg.predict(X.drop_duplicates())

from sklearn import tree

arvore = tree.DecisionTreeRegressor(random_state=42)
arvore.fit(X,y)
predict_arvore = arvore.predict(X.drop_duplicates())


arvore2 = tree.DecisionTreeRegressor(random_state=42, max_depth= 2)
arvore2.fit(X,y)
predict_arvore2 = arvore2.predict(X.drop_duplicates())

#%%

import matplotlib.pyplot as plt

plt.plot(X['cerveja'],y, 'o')
plt.grid(True)
plt.title('Cerveja vs Nota')
plt.xlabel('Qtd cervejas')
plt.ylabel('Nota')

plt.plot(X.drop_duplicates()['cerveja'],predict)
plt.plot(X.drop_duplicates()['cerveja'],predict_arvore)
plt.plot(X.drop_duplicates()['cerveja'],predict_arvore2)

plt.legend(['Pontos observados', f'y = {a:.3f} + {b:.3f}X','Arvore','Depth = 2'])

