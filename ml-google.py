#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %%
base_casa = pd.read_csv(r'\workspace\data\ml-google.csv', sep=';')
base_casa = base_casa.drop(columns=['origem', 'taxa_engajamento', 'taxa_eventos_principais'])
# %%
#sns.pairplot(base_casa)
# %%
base_casa.columns
# %%
'''sns.pairplot(base_casa, x_vars=['total_usuarios', 'sessoes', 'sessoes_engajadas',
       'contagem'], y_vars='eventos_principais')'''
# %%
base_casa.corr()
# %%
X = base_casa[['total_usuarios', 'sessoes', 'sessoes_engajadas',
       'contagem']]

Y = base_casa[['eventos_principais']]
# %%
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size= 0.7, test_size= 0.3, random_state = 42)
# %%
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# %%
from sklearn.linear_model import LinearRegression
# %%
lm = LinearRegression()
# %%
lm.fit(X_train, Y_train)
# %%
y_pred = lm.predict(X_test)
# %%
from sklearn.metrics import r2_score
r = r2_score(Y_test, y_pred)
# %%
print("r_quadrado:", r)
# %%
c = [i for i in range(1, 51, 1)]
fig = plt.figure(figsize=(10,8))
plt.plot(c, Y_test, color="blue")
plt.plot(c, y_pred, color="red")
plt.xlabel("index")
plt.ylabel("evento_principal")
# %%
total_usuarios = 2000
sessoes = 2200
sessoes_engajadas = 500
contagem = 8000
entrada = [[total_usuarios, sessoes, sessoes_engajadas, contagem]]
lm.predict(entrada)[0]
# %%
