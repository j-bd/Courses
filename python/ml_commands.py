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

from pandas_profiling import ProfileReport
profile = ProfileReport(df, title='Pandas Profiling Report', html={'style':{'full_width':True}})
profile.to_file(output_file="your_report.html")

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
# DATA CLEANING
# =============================================================================

df['class'].replace(["Iris-setossa","versicolor"], ["Iris-setosa","Iris-versicolor"], inplace=True)


# =============================================================================
# Graph
# =============================================================================
"si les variables explicatives sont toutes quantitatives et les colonnes peu nombreuses"
sns.pairplot(df, hue='target', size=2.5)



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


'''Pearson/Spearman'''
from scipy.stats import  pearsonr, spearmanr


def correlation_tests(dataframe, x1, x2, alpha = 0.05, test = 'pearson'):
  if test not in ['pearson', 'spearman'] :
    print('Error - please specify a correct test : Pearson, Spearman')

  if test == 'pearson' :
    pearson_corr, pearson_p_value = pearsonr(dataframe[x1], dataframe[x2])
    print('\nPerson correlation :')
    print('====================')
    print('correlation : {}, p_value : {:.5E}'.format(pearson_corr, pearson_p_value))
    if pearson_p_value < 0.05 :
      print('Person correlation results : Dependent (reject H0)')
    else :
      print('Person correlation results : Independent (fail to reject H0). No evidence of a linear relatioship.')

  if test == 'spearman' :
    spearman_corr, spearman_p_value = spearmanr(dataframe[x1], dataframe[x2])
    print('\nSpearman correlation :')
    print('======================')
    print('correlation : {}, p_value : {:.5E}'.format(spearman_corr, spearman_p_value))
    if spearman_p_value < 0.05 :
      print('Spearman correlation results : Dependent (reject H0)')
    else :
      print('Spearman correlation results : Independent (fail to reject H0). No evidence of a monotonic relatioship')


'''Chie2'''
from scipy.stats import chi2_contingency,chi2

def chi_square_independence_test(dataframe, x1, x2, alpha ) :
  '''
  Function that conductes Pearson’s Chi-Squared of independance

  Arguments
  =========
  dataframe : pandas dataframe
  x1 : str. First variable
  x2 : str. Second variable
  alpha : float, significance treshold

  Returns
  =======
  stat, p, dof
  if dependent or not
  '''


  print('\nPearson’s Chi-Squared Test :')
  print('============================')
  stat, p, dof, expected = chi2_contingency(pd.crosstab(dataframe[x1], dataframe[x2]).values)
  print('stat : {}, p : {:.5E}, dof : {}'.format(stat,p, dof))

  # interpret test-statistic
  if p <= alpha:
    print('Dependent (reject H0)')
  else:
      print('Independent (fail to reject H0)')


'''Kruskal-Wallis'''
#Ho:same distributions -> independant

from scipy.stats import kruskal

df_kw = df[['va1', 'va2']].dropna()

df_kw_1 = df_kw[df_kw['var1'] == 'A'].var2
df_kw_2 = df_kw[df_kw['var2'] == 'B'].var2
df_kw_3 = df_kw[df_kw['var3'] == 'C'].var2
df_kw_4 = df_kw[df_kw['var4'] == 'D'].var2

stat, p = kruskal(df_kw_1, df_kw_2, df_kw_3, df_kw_4)
print('Statistics=%.3f, p=%.3f' % (stat, p))

#interpret
alpha = 0.05

if p > alpha:
    print('Same distribution (fail to reject H0) -> independant')
else:
    print('Different distributions (reject H0) -> Dependant')

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

"suppression de ligne"
df.dropna(axis=0, inplace=True)


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
from sklearn.preprocessing import StandardScaler
cells_array_std = StandardScaler().fit_transform(cells_array)
pca = PCA(n_components=2)
pca_2d_std = pca.fit_transform(cells_array_std)


tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2000)
tsne_2d_std = tsne.fit_transform(cells_array_std)



# =============================================================================
# Exemple
# =============================================================================
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[1,1], random_state=1)

# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
# fit a model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(trainX, trainy)

# predict probabilities
probs = model.predict_proba(testX)

# keep probabilities for the positive outcome only
probs = probs[:, 1]



# =============================================================================
# Evaluation de l algorithme
# =============================================================================
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot


'to compute the AUC and plot the ROC curve '
# calculate AUC
model_auc = roc_auc_score(testy, probs)
print('AUC: %.3f' % model_auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(testy, probs)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
pyplot.show()



"""
Precision Recall curve
F1, AUC and AP scores
skilful models are generally above 0.5.
"""
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score

# predict probabilities
probs = model.predict_proba(testX)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model.predict(testX)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(testy, probs)
# calculate F1 score
f1 = f1_score(testy, yhat)
# calculate precision-recall AUC !!!Different from ROC-AUC!!!
pr_auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(testy, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, pr_auc, ap))
# plot no skill
pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
pyplot.plot(recall, precision, marker='.')
# show the plot
pyplot.show()


