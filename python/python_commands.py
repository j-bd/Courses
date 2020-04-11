#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 08:37:50 2020

@author: j-bd
"""


# =============================================================================
# NUMPY COMMAND
# =============================================================================

import numpy as np

"Return evenly spaced numbers over a specified interval."
np.linspace(2.0, 3.0, num=5)

array([2.  , 2.25, 2.5 , 2.75, 3.  ])

"Return the indices of the bins to which each value in input array belongs."
x = np.array([0.2, 6.4, 3.0, 1.6])
bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
inds = np.digitize(x, bins)
inds

array([1, 4, 3, 2])

for n in range(x.size):
    print(bins[inds[n]-1], "<=", x[n], "<", bins[inds[n]])

0.0 <= 0.2 < 1.0
4.0 <= 6.4 < 10.0
2.5 <= 3.0 < 4.0
1.0 <= 1.6 < 2.5

"Produits terme à terme entre les deux matrices"
a=np.array([[2,3,4],[1,5,6]])
b=np.array([[1,2,3]])

a*b

"Calcul de déterminant de matrice carrée"
from numpy.linalg import det
a = np.array([[1, 2], [3, 4]])
det(a)
# Si le déterminant est nul la matrice n'est pas inversible

"Construire une matrice symétrique"
x = np.array([[1.0,5.0,4,7,9,2,3,5]])  # il faut définir un vecteur ligne (matrice 1 x n)
M = x.T.dot(x)

"Decomposition LU"
import scipy.linalg as sl
A = np.random.random(size=(4,4))
(P2,L2,U2) = sl.lu(A)

"Résolution d’un système d’équations linéaires"
a = np.array([[3,1], [1,2]]) #coefficient des x
b = np.array([9,8]) #les y

x = np.linalg.solve(a, b) # resolution
np.allclose(np.dot(a, x), b) # verification

"Calcul de valeurs propres et vecteurs propres"
from numpy.linalg import eig
A = np.array([[ 1, 1, -2 ], [-1, 2, 1], [0, 1, -1]])
D, V = eig(A)
#verifiction : AV = DV où * est le produit scalaire matriciel

"Visualisation de variable en 3D"
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.random.standard_normal(100)
y = np.random.standard_normal(100)
z = np.random.standard_normal(100)
c = np.random.standard_normal(100)

img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
fig.colorbar(img)
plt.show()

"Matrice identité"
np.eye(3)

"Transposée"
a = np.array([[1, 2, 3], [4, 5, 6]])
a.T



# =============================================================================
# PANDAS COMMAND
# =============================================================================

import pandas as pd

# OVERVIEW

url = "http://www.lpsm.paris/pageperso/bousquet/yotta/data/villes-belges.csv"
df = pd.read_csv(url,index_col='Commune')

"Produire un résumé global des donnée"
print(df.info())
df.shape

"produire un descriptif plus fin des données"
print(df.describe(include='all'))

"affichage"
pd.options.display.max_columns = 10
pd.options.display.max_rows = 10
print(df.head()) #haut du tableau
print(df.tail()) #bas du tableau

"Enumeration des colonnes"
df.columns

"Type de chaque colonne"
print(df.dtypes)

"Statistiques descriptives par colonne"
df.describe()

"Valeur nulle par colonne"
print(df.isnull().sum(axis=0))


"final block for global overview"
df.describe()
df.info()
df.columns
df.isnull().sum(axis=0)
df.head()
df.nunique(dropna = False) #nb unique value for each column


'Vérifier si des lignes en doublon existent (les valeurs de toutes les colonnes sont les mêmes dans ce cas)'
print(len(df.iloc[pd.Series(df.duplicated(keep = False)).loc[lambda x : x==True].index]))





# PREPROCESS

"Accédez simultanément aux colonnes col1 et col2"
df[['col1','col2']]
"Calculez les moyennes et comptages pour une colonne"
print(df['col1'].mean())
print(df['col2'].value_counts())

"On souhaite appliquer à toutes les variables un ensemble de mêmes fonctions"
"(par exemple la moyenne)"

def operation(x):
    return (x.mean())
resultat = df.select_dtypes(exclude=['object']).apply(operation,axis=0)
print(resultat)

"tri généralisé aux DataFrame"
df.sort_values(by='col1')

"scission des données"
g = df.groupby("col_1")
g.get_group("val")

"Calculer différentes fonctions à des variables via la fonction .agg(),"
"telle la moyenne et l'écart-type"
g[['col_2','col_3']].agg([pd.Series.mean,pd.Series.std])


"Créer une colonne 'rented' représentant l'addition de 2 colonnes que vous supprimez"
columns_to_exclude = ['casual', 'registered']
df['rented'] = df['casual'] + df['registered']
df.drop(columns_to_exclude, axis=1, inplace=True)


" Transformer les variables qui devraient être catégorielles en variables de type catégoriel"
categoryVariableList = ["season","weather","holiday","workingday"]
for var in categoryVariableList:
    df[var] = df[var].astype("category")


" Transformer les variables temporelles (datetime et date) en variables de type date ou datetime"
import calendar
from datetime import datetime
df["date"] = df.datetime.apply(lambda x : x.split(" ")[0])
df["hour"] = df.datetime.apply(lambda x : x.split()[1].split(":")[0])
df["weekday"] = df.date.apply(
    lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()]
)
df["month"] = df.date.apply(
    lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month]
)


df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df["weekday"] = df['datetime'].dt.day_name()
df["month"] = df['datetime'].dt.month_name()

"Vérifier que la colonne datetime consitue bien un identifiant de ligne"
print(df.datetime.nunique() == df.shape[0])

"Calculer le nombre total des locations initiées par les utilisateurs entre le 12-03-2011 et le 15-03-2011"
df.loc[(df['date']>='2011-03-12') & (df['date']<='2011-03-15')].rented.sum()


# GRAPH
import seaborn as sn
"Tracé d'histogramme et de densité"
df.hist(column='col1')
df.col1.plot.hist()#autre solution

"histogramme des fréquences"
df.hist(column='col1',density=True)

"estimateur à noyaux de la densité avec la fonction"
df['col1'].plot.kde() # voir egalement https://scikit-learn.org/stable/modules/density.html

"histogramme de la distribution des revenus moyens par commune selon les provinces"
df.hist(column='averageincome',by="Province",figsize=(10, 10))

"commande utiles pour varier le tracé des histogrammes:"
"http://www.python-simple.com/python-matplotlib/histogram.php"


"""
Visualiser :

    La distribution mensuelle des locations par mois
    La distribution horaire des locations en fonction des saisons
    La distribution horaire des locations en fonction des jours de la semaine
"""
fig,(ax1,ax2,ax3)= plt.subplots(nrows=3)
fig.set_size_inches(12,20)
sortOrder = ["January","February","March","April","May","June","July","August","September","October","November","December"]
hueOrder = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

monthAggregated = pd.DataFrame(df.groupby("month")["rented"].mean()).reset_index()
monthSorted = monthAggregated.sort_values(by="rented",ascending=False)
sn.barplot(data=monthSorted,x="month",y="rented",ax=ax1,order=sortOrder)
ax1.set(xlabel='Month', ylabel='Avearage Rented',title="Average Rount By Month")

hourAggregated = pd.DataFrame(df.groupby(["hour","season"],sort=True)["rented"].mean()).reset_index()
sn.pointplot(
    x=hourAggregated["hour"], y=hourAggregated["rented"],hue=hourAggregated["season"],
    data=hourAggregated, join=True,ax=ax2
)
ax2.set(
    xlabel='Hour Of The Day', ylabel='Rented',title="Average By Hour Of The Day Across Season",label='big'
)

hourAggregated = pd.DataFrame(df.groupby(["hour","weekday"],sort=True)["rented"].mean()).reset_index()
sn.pointplot(
    x=hourAggregated["hour"], y=hourAggregated["rented"],hue=hourAggregated["weekday"],
    hue_order=hueOrder, data=hourAggregated, join=True,ax=ax3
)
ax3.set(
    xlabel='Hour Of The Day', ylabel='Rented',title="Average By Hour Of The Day Across Weekdays",
    label='big'
)


"Etudier les relations entre les différentes variables. Quelles variables sont "
"très corrélées ? Visualiser ces corrélations grâce à la fonction heatmap de seaborn"
quantitatives_cols = ['temp', 'atemp', 'humidity', 'windspeed', 'rented']
sn.pairplot(
    df[quantitatives_cols], diag_kind="hist",
    plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'}, height = 4
)

corrMatt = df[quantitatives_cols].corr()
sn.heatmap(corrMatt,vmax=.8, square=True,annot=True)


