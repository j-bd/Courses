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


