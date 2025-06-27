# %%
import pandas as pd

# CORREÇÃO: Usar a barra normal (/) no caminho do arquivo para maior compatibilidade
# entre sistemas operacionais (Windows, macOS, Linux).
data = pd.read_excel('data/dados_frutas.xlsx')
data.head()

# %%
from sklearn import tree

# CORREÇÃO: Renomeie a variável do classificador para não sobrescrever o módulo 'tree'.
# É uma prática comum usar 'clf' (de classifier) ou 'model'.
clf = tree.DecisionTreeClassifier(random_state=42)

y = data['Fruta']

# CORREÇÃO: Corrigido o erro de digitação em 'characteristics'.
characteristics = ['Arredondada', 'Suculenta', 'Vermelha', 'Doce']
x = data[characteristics]

# %%
# Treinando o modelo com a variável correta ('clf')
clf.fit(x, y)

# Fazendo a previsão com o modelo correto
# O resultado será um array com a fruta prevista, por ex: array(['Maçã'], dtype=object)
clf.predict([[1, 1, 1, 1]])

# %%
import matplotlib.pyplot as plt # CORREÇÃO: Importa-se o 'pyplot' para plotagem.

# CORREÇÃO:
# 1. A função 'plot_tree' pertence ao módulo 'tree' original.
# 2. O primeiro argumento deve ser o classificador treinado ('clf').
# 3. Use as variáveis com os nomes corrigidos.
# 4. Adicionado plt.figure() para definir um bom tamanho para a imagem.

plt.figure(figsize=(15, 10)) # Define o tamanho da figura para melhor visualização
tree.plot_tree(
    clf,
    feature_names=characteristics,
    class_names=clf.classes_,
    filled=True, # Adiciona cores aos nós para indicar a classe majoritária
    rounded=True # Deixa os nós com cantos arredondados
)
plt.show() # Exibe o gráfico gerado

# %%

proba = clf.predict_proba([[1,1,1,1]])[0]

pd.Series(proba, index = clf.classes_)