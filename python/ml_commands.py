#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:32:54 2020

@author: j-bd
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("path")



# =============================================================================
# Global overview
# =============================================================================
df.describe()
df.info()
df.columns
df.isnull().sum(axis=0)
df.head()
df.nunique(dropna = False) #nb unique value for each column



# =============================================================================
# OUTLIERS
# =============================================================================

" Catégoriser les outliers de la population du label ==> Proposer un petit script"
" pour visualiser les outliers sur la variable cible."
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)
sns.boxplot(data=df,y="rented",orient="v",ax=axes[0][0])
sns.boxplot(data=df,y="rented",x="season",orient="v",ax=axes[0][1])
sns.boxplot(data=df,y="rented",x="hour",orient="v",ax=axes[1][0])
sns.boxplot(data=df,y="rented",x="workingday",orient="v",ax=axes[1][1])

axes[0][0].set(ylabel='rented',title="Box Plot On Count")
axes[0][1].set(xlabel='Season', ylabel='rented',title="Box Plot On Count Across Season")
axes[1][0].set(xlabel='Hour Of The Day', ylabel='rented',title="Box Plot On Count Across Hour Of The Day")
axes[1][1].set(xlabel='Working Day', ylabel='rented',title="Box Plot On Count Across Working Day")

"méthode pour filtrer ces données anormales (Par exemple en utilisant la variance"
" de la distribution). Combien d'outliers à-t-on filtré ?:"

dailyDataWithoutOutliers = df[
    np.abs(df["rented"] - df["rented"].mean()) <= (3 * df["rented"].std())
]

print ("Shape of the dataframe before ouliers removal: ",df.shape)
print ("Shape of the dataframe after ouliers removal: ",dailyDataWithoutOutliers.shape)




# =============================================================================
# Graph
# =============================================================================

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





# =============================================================================
# Statistics
# =============================================================================

"Loi normale"
from scipy.stats import norm
import matplotlib.pyplot as plt

def loi_normale(x,mu = 0 ,sigma = 1):
    return norm.pdf(x,loc = mu, scale=sigma)

x = np.arange(-10,10,0.1)
y = loi_normale(x)
plt.plot(x,y)


# =============================================================================
# MISSING VALUES
# =============================================================================
missing_values_pourc = df.isnull().sum()/len(df)*100
missing_values_pourc[missing_values_pourc>0].sort_values(ascending = False)
'Drop'
col_list = [idx for idx, val in missing_values_pourc.iteritems() if val >= 60]
df = df.drop(columns=col_list)
'imputer les valeurs manquantes avec le mode'
data_columns = df.columns
for col in data_columns:
    df[col].fillna(df[col].mode()[0],inplace = True)


# =============================================================================
# SELECTION DE VARIABLES
# =============================================================================

"Calculer la variance pour l'ensemble des variables quantitatives"
df.var().sort_values(ascending=True)
"Gardez uniquement les variables avec une variance supérieure ou égale à 10"
numeric_variables = df.select_dtypes(include=[np.number])
numeric_variables_variance = numeric_variables.var()
numeric_columns = numeric_variables.columns
columns_LVF = [ ]
for i in range(0,len(numeric_columns)):
    if numeric_variables_variance[i]>=10 :
        columns_LVF.append(numeric_columns[i])

import seaborn as sns

"Filtre à corrélation élevée"
applications_without_target = df.drop('TARGET', 1)
f, ax = plt.subplots(figsize = (14,12))
plt.title('Correlation of Features - HeatMap',y=1,size=16)
sns.heatmap(applications_without_target.corr(),square = True,  vmax=0.8)


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
"PCA"
'Convertir le dataframe en tableau (type arrays) puis appliquer PCA à 2 composantes.'
'df without target variable'

cells_array = df.values
pca = PCA(n_components=2)
pca_2d = pca.fit_transform(cells_array)

" Appliquer-t-SNE avec les paramètres suivants : n_components=2, verbose=1, perplexity=40, n_iter=2000"
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2000)
tsne_2d = tsne.fit_transform(cells_array)

"Visualiser le résultat de t-SNE et de PCA côte-à-côte en mettant la variable cible en couleur"
plt.figure(figsize = (16,11))
plt.subplot(121)
plt.scatter(pca_2d[:,0],pca_2d[:,1], c = cells_target,
            cmap = "coolwarm", edgecolor = "None", alpha=0.35)
plt.colorbar()
plt.title('PCA Scatter Plot')


plt.subplot(122)

plt.scatter(tsne_2d[:,0],tsne_2d[:,1],  c = cells_target,
            cmap = "coolwarm", edgecolor = "None", alpha=0.35)
plt.colorbar()
plt.title('TSNE Scatter Plot')
plt.show()

"Standardiser l'ensemble des variables"
cells_array_std = StandardScaler().fit_transform(cells_array)
pca = PCA(n_components=2)
pca_2d_std = pca.fit_transform(cells_array_std)


tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2000)
tsne_2d_std = tsne.fit_transform(cells_array_std)
