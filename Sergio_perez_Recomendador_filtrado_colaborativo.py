
# coding: utf-8

# # Recomendador de peliculas Movieslens 
# Sergio David Pérez Navarro

# Vamos a emplear el dataset de Movieslens
# This data set consists of:
# 	* 100,000 ratings (1-5) from 943 users on 1682 movies. 
# 	* Each user has rated at least 20 movies. 
#         * Simple demographic info for the users (age, gender, occupation, zip)
#         
# Para crear el recomendador vamos a emplear un método muy común que fue explicado en clase. El denominado filtro colaborativo. Este método consiste en al comparación de los usuarios entre si para encontrar una similitud entre estos mediante el uso de una distancia. Podemos construir una matriz que contiene todos los usuarios y todas las películas con su correspondiente rating.
# En este caso dicha matriz se llama $\textbf{X_train_mat}$. A continuación podemos cálcular la matriz de distancia en cada uno de los ejes.

# In[11]:

path = "E:\\Documentos\\Master\\movies\\ml-100k\\" #Dataset de 100k usuarios

import pandas as pd #version 19.2
import matplotlib.pyplot as plt #version 2.0
import numpy as np #version 1.11.3
import seaborn as sns #version 0.7.1
from scipy.spatial.distance import pdist #version 18.1
from sklearn.preprocessing import MinMaxScaler #version 18.1
from sklearn.model_selection import train_test_split#version 18.1
from scipy import sparse #version 18.1
from sklearn.decomposition import TruncatedSVD #version 18.1
from sklearn.metrics.pairwise import pairwise_distances #version 18.1
get_ipython().magic('matplotlib notebook')


# In[14]:

df_movies = pd.read_table(path + "u.data", names = ["user_id","item_id","rating","timestamp"])
df_users = pd.read_table(path + "u.user", names = ["user_id","age","gender","occupation", "zip_code"], sep="|", index_col="user_id")
df_item = pd.read_table(path + "u.item", sep="\|",names = ["item_id", "titulo","release_date","video_release_date","IMDb_URL","unknown","Action", "Adventure","Animation","Children","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance", "Sci-Fi","Thriller","War","Western"])
df_users.zip_code = pd.to_numeric(df_users.zip_code, errors=np.nan)
df_users = df_users.dropna()


# In[4]:

X_train, X_test = train_test_split(df_movies, test_size=0.33, random_state=42)
X_train_mat = X_train.pivot(index="item_id", columns="user_id", values="rating").fillna(0)
X_test_mat = X_test.pivot(index="item_id", columns="user_id", values="rating").fillna(0)

X_train_mat = X_train_mat.as_matrix()
X_test_mat = X_test_mat.as_matrix()

train_user_ids = X_train.user_id.values
train_item_ids = X_train.item_id.values

user_similarity = pairwise_distances(X_train_mat, metric='cosine')
item_similarity = pairwise_distances(X_train_mat.T, metric='cosine')
user_similarity_euclidean = pairwise_distances(X_train_mat, metric='euclidean')
item_similarity_euclidean = pairwise_distances(X_train_mat.T, metric='euclidean')


# A cotinuación, a modo de curiosidad vamos a realizar una representación de la matriz de similitud.

# In[10]:

plt.figure(dpi=150)
sns.heatmap(1-user_similarity, xticklabels=100, yticklabels=100)
plt.xticks(rotation = 90)
plt.yticks(rotation = 0)
plt.title("Similitud entre usuarios")
plt.show()

plt.figure(dpi=150)
sns.heatmap(1-item_similarity, xticklabels=100, yticklabels=100)
plt.xticks(rotation = 90)
plt.yticks(rotation = 0)
plt.title("Similitud entre películas")
plt.show()


# En el proceso de investigación sobre filtros colaborativos se visitaron númerosas páginas y documentos explicando como realizar la predicción. Existen diversas formas de proceder, pero uno de los factores comunes cuanto tratamos este tipo de casos es que no todos los usuarios puntuan de la misma forma. Existen usuarios que puntuan de forma más generosa y otros que puntuan de forma más estricta a pesar de que pueden (o no ) tener los mismos gustos. Con el fin de compensar este fenómeno se emplea la siguiente función de predicción. Esta tiene de partícular que se emplean las diferencias medias relativas en rating y no simplemente el  valor absoluto. Una vez se dispone de estas medias se emplea el producto escalar (en la práctica es una métrica coseno).
# Además la función puede ser empleada para con usuarios recomendar películas o con películas recomendar usuarios (las operaciones son exactamente las mismas, la diferencia es que las películas no presentan el fenómeno antes mencionado, lógicamente).

# In[6]:

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
    return pred


# In[7]:

item_prediction = predict(X_train_mat, item_similarity, type='item')
user_prediction = predict(X_train_mat, user_similarity, type='user')

from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

print("Distancia Coseno")
print('Root mean squared error Usuarios: ' + str(rmse(user_prediction, X_test_mat)))
print('Root mean squared error Peliculas: ' + str(rmse(item_prediction, X_test_mat)))

item_prediction_euclidean = predict(X_train_mat, item_similarity_euclidean, type='item')
user_prediction_euclidean = predict(X_train_mat, user_similarity_euclidean, type='user')
print("Distancia euclidea")
print('Root mean squared error Usuarios: ' + str(rmse(user_prediction_euclidean, X_test_mat)))
print('Root mean squared error Peliculas: ' + str(rmse(item_prediction_euclidean, X_test_mat)))


# In[8]:

pelis = df_item.item_id.isin(train_item_ids[np.argsort(item_prediction, axis = 0)[0,:]])
print("Distancia Coseno:\n",df_item[pelis].titulo.head(),"\n") #Veamos los nombres de las peliculas recomendadas para todos los usuarios.
pelis_euclidean = df_item.item_id.isin(train_item_ids[np.argsort(item_prediction_euclidean, axis = 0)[0,:]])
print("Distancia euclidea:\n",df_item[pelis_euclidean].titulo.head()) #Veamos los nombres de las peliculas recomendadas para todos los usuarios.


# Con el fin de evitar recomendar constantemente las mismas películas a usuarios que ya las han visto, vamos a descartar estos casos.

# In[9]:

item_prediction2 = np.copy(item_prediction) #shape = (1624 items, 943 users)
item_prediction2_euclidean = np.copy(item_prediction_euclidean) #shape = (1624 items, 943 users)
 
item_prediction2[X_train_mat > 0] = 0 #Reducimos el score de recomendación de ver una pelicula puntuada a cero
item_prediction2_euclidean[X_train_mat > 0] = 0 #Reducimos el score de recomendación de ver una pelicula puntuada a cero


pelis = df_item.item_id.isin(train_item_ids[np.argsort(item_prediction2, axis = 0)[0,:]])
print("Distancia Coseno:\n",df_item[pelis].titulo.head(),"\n") #Veamos los nombres de las peliculas recomendadas para todos los usuarios.
pelis_euclidean = df_item.item_id.isin(train_item_ids[np.argsort(item_prediction2_euclidean, axis = 0)[0,:]])
print("Distancia euclidea:\n",df_item[pelis_euclidean].titulo.head()) #Veamos los nombres de las peliculas recomendadas para todos los usuarios.


# Como curiosidad hay que mencionar que a pesar de que prácticamente el 100% de los métodos consultados durante la creación del ejercicio recomiendan emplear la distancia coseno, la distancia euclidea aquí empleada por mera completitud da menores valores de RMSE.
