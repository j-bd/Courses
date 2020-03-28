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
import seaborn as sns

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

"Accédez simultanément aux colonnes col1 et col2"
df[['col1','col2']]
"Calculez les moyennes et comptages pour une colonne"
print(df['col1'].mean())
print(df['col2'].value_counts())

"On souhaite appliquer à toutes les variables un ensemble de mêmes fonctions"
"(par exemple la moyenne)"
import numpy as np
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


"Tracé de la fonction de répartition empirique"
from statsmodels.distributions.empirical_distribution import ECDF

sample = df['averageincome']# toute la Belgique

prov_Anvers = df.loc[df['Province']=='Anv.',:]
sample2 = prov_Anvers['averageincome'] # province d'Anvers seulement

ecdf = ECDF(sample)
x = np.linspace(min(sample), max(sample)) # trace une grille régulière de valeurs entre 2 bornes
y = ecdf(x)

ecdf2 = ECDF(sample2)
y2 = ecdf2(x)

figure = plt.figure(figsize = (10, 6))
plt.gcf().subplots_adjust(left = 0.2, bottom = 0.2, right = 0.9, top = 0.9, wspace = 0.5, hspace = 0)
plt.subplot(1, 2, 1)
plt.step(x, y)
plt.xlabel('Ensemble de la Belgique')
plt.subplot(1, 2, 2)
plt.step(x, y2)
plt.xlabel('Région d''Anvers')
plt.suptitle('Fonction de répartition empirique : revenus médians')
plt.show()

"fonctions dans la librairie PyLab pour représenter des courbes superposées"
# https://courspython.com/introduction-courbes.html


"QQ-plot"
import scipy.stats as stats
import pylab

# Lister les provinces
prov = np.unique(df.Province)

# On fait une boucle sur la liste des provinces
for p in prov:
    data = df[df['Province']==p]
    x = data['averageincome']
    figure = plt.figure(figsize = (5, 3))
    stats.probplot(x,dist="norm",plot=pylab)
    pylab.title(str(p))
    pylab.show()

"Calcul de quantile empirique"
np.percentile(df['averageincome'], [1/4, 1/2,3/4])

"Calcul de variance et dispersion"
variance = np.var(df['medianincome'])
ecart_type = np.std(df['medianincome'])

# Calcul de l'écart inter-quartiles
quantiles = np.percentile(df['averageincome'],[25,75])
IQR = quantiles[1]- quantiles[0]


"Boîtes à moustaches (boxplots) et violin plots"
fig1, ax1 = plt.subplots()
ax1.set_title('Boîte à moustaches basique')
ax1.boxplot(df['medianincome'])
fig2, ax2 = plt.subplots()
ax2.set_title('Boîte à moustaches "crantée"')
ax2.boxplot(df['medianincome'], notch=True)

"représenter l'étalement des revenus par province "
df.boxplot(column='medianincome',by="Province",figsize=(10, 10))
#violin plot
g = sns.catplot(x='Province',y='medianincome',kind='violin',data=df,size=10)
sns.swarmplot(x='Province',y='medianincome',color='k',size=3,data=df,ax=g.ax)
#violin plot
figure = plt.figure(figsize = (10, 10))
sns.set(style="whitegrid")
ax = sns.violinplot(x="Province",y='medianincome',data=df,inner="quartile")
