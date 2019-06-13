# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from savejupyterdata import load, save
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

import os
import time
import re
import numba

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# # Data preclean

# %%
fd_pos = pd.read_csv('pos_definitions.csv')
fd_neg = pd.read_csv('neg_definitions.csv')
into_pos = pd.read_csv('pos_featureValues.csv')
into_neg = pd.read_csv('neg_featureValues.csv')


fd_pos.rename(columns={'Unnamed: 0':'feature_index'}, inplace=True)
fd_neg.rename(columns={'Unnamed: 0':'feature_index'}, inplace=True)
into_pos.rename(columns={'Unnamed: 0':'feature_index'}, inplace=True)
into_neg.rename(columns={'Unnamed: 0':'feature_index'}, inplace=True)

def remove_first_two_num(col):
    if re.match('^[0-9]{2}.*.mzXML', col):
        return col[2:-6].replace('_pos', '').replace('_neg', '')
    elif 'QC' in col:
        return col[:-6].replace('_pos', '').replace('_neg', '')
    else:
        return col.replace('_pos', '').replace('_neg', '')
    
into_pos.rename(columns=remove_first_two_num, inplace=True)
into_neg.rename(columns=remove_first_two_num, inplace=True)


# %%
def get_index(cols):
    qcs = []
    filters = []
    tests = []
    for i, col in enumerate(cols):
        if col.startswith('QC'):
            qcs.append(i)
        elif col.startswith('Fi'):
            filters.append(i)
        else:
            tests.append(i)
    return qcs, filters, tests


# %%
def classify(col):
    if 'cond' in col:
        return col[:-1]
    else:
        return re.match('(.*)_.*', col)[1]


# %%
def draw_pca(df, fi=False):
    
    qcs, filters, tests = get_index(df.columns)
    if not fi:
        df = df[df.columns^df.columns[filters]]
    hues = []
    for col in df.columns:
        hues.append(classify(col))
        
    cols = df.columns
    data = StandardScaler().fit_transform(df.values.T)
    pca = PCA(n_components=8, whiten=True).fit_transform(data)
    #pca = PCA(n_components=8).fit_transform(df.values.T)
    pca_df = pd.DataFrame(pca)
    pca_df['cate'] = hues
    pca_df.index=cols
    colormap = {'QC':'grey', 'QC_cond':'lightgrey',
                'A': 'coral', 'B': 'orange', 'C': 'forestgreen',
                'D': 'yellow', 'E': 'crimson', 'F': 'royalblue'}
    fig, ax = plt.subplots(1,1, figsize=(10,6))
    sns.scatterplot(0, 1, hue='cate', data=pca_df, s=300, palette=colormap, alpha=.8, ax=ax)
    
    pca_df = pca_df.reindex(df.columns[1:]).dropna()

    for i, ind in enumerate(pca_df.index):
        ax.text(pca_df.loc[ind, 0], pca_df.loc[ind, 1]+.1, i+1, color='maroon', fontsize=14)
        
    ax.legend(fontsize=18, bbox_to_anchor=[1,1,0,0])
    plt.tight_layout()
    return pca_df


# %%
class MoreThanOneIndexCol(Exception):
    pass


# %% [markdown]
# # Filter and EDA

# %%
index_neg_qc, index_neg_filter, index_neg_tests = get_index(into_neg.columns)
index_pos_qc, index_pos_filter, index_pos_tests = get_index(into_pos.columns)

# %%
filters_pos = into_pos.iloc[:, index_pos_filter]
filters_pos = filters_pos.T.sort_index().T

# %%
from collections import defaultdict


# %%
class ChromatographData():
    """
    Objects contain data and transform method
    """
    def __init__(self, data, polarity='postive', feature_index_col='parse'):
        """
        :data: pd.DataFrame
        :polarity: 'positive' or 'negative'
        :feature_index_col: col identifier if there is
        """
        #self._polartity
        if not any([polarity.lower() in two for two in ['positive', 'negative']]):
            raise ValueError(f'invalid polarity parameter!{polarity}')
        if polarity.lower() in 'positive':
            self._polarity = 'Positive'
        else:
            self._polarity = 'Negative'
        
        #self._data and self._index
        if feature_index_col =='parse':
            self._index = data.loc[:, data.dtypes==np.object]
            if self._index.shape[1]>1:
                raise MoreThanOneIndexCol(
                    f'data has {self._index.shape[1]} object columns! (must be 1)')
            self._data = data.loc[:, data.dtypes!=np.object]
            
        elif isinstance(feature_index_col, int):
            self._index = pd.DataFrame(data.iloc[:, feature_index_col]).rename(
                columns={0:data.columns[feature_index_col]})
            self._data = data.loc[:, data.columns != data.columns[feature_index_col]]
        
        elif isinstance(feature_index_col, str):
            self._index = pd.DataFrame(data.loc[:, feature_index_col]).rename(
                columns={0:feature_index_col})
            self._data = data.loc[:, data.columns != feature_index_col]
            
        else:
            raise TypeError('feature_index_col must be "parse", int or str!')
        
        #get indices list about QCs, filters and test samples
        self._qc_indices, self._filter_indices, self._test_indices = self._get_index()
        self._categories = self._classify()
        
    def __repr__(self):
        """
        print Data status
        include data shape, process history
        """
        return(f'''Chromatograph data:\n
        data shape {self._data.shape}''')
    
    def _get_index(self):
        qcs = []
        filters = []
        tests = []
        for i, col in enumerate(self._data.columns):
            if col.startswith('QC'):
                qcs.append(i)
            elif col.startswith('Fi'):
                filters.append(i)
            else:
                tests.append(i)
        return [pd.Index(indices) for indices in [qcs, filters, tests]]
    
    @property
    def df(self):
        return self._data
    
    def _classify(self):
        cates = defaultdict(list)
        for i, col in enumerate(self._data.columns):
            if 'cond' in col:
                cates[col[:-1]].append(i)
            else:
                cates[re.match('(.*)_.*', col)[1]].append(i)
        return cates
    
    def deal_nan(self, method=None, value=None):
        if method is None and value:
            self._data.fillna(value)
        else:
            pass
        
    def fill_by_categories(self, method='median'):
        for cate, raw in self._categories.items():
            self._data.iloc[:, raw] = self._data.iloc[:, raw].apply(lambda x: x.fillna(x.median()), axis=1)
        self.fill_across_all_samples()
          
    def fill_fast(self):
        data = self._data.values
        return self.fill_helper(data)
    
    @numba.njit
    def fill_helper(self, temp, dic):
        
        for cate, raw in dic.items():
            raw_median = np.apply_along_axis(lambda x: np.nanmedian(x), 1, temp[:, raw])
            for i, row in enumerate(temp[:, raw]):
                temp_row = temp[row, raw]
                temp_row[np.isnan(temp_row)] = raw_median[i]
                temp[row, raw] = temp_row
        return temp

    def fill_across_all_samples(self):
        self._data = self._data.apply(lambda x: x.fillna(x.min())/(3**0.5), axis=1)


# %%
a = ChromatographData(into_pos, 'positive')

# %%
t = a.df.values.copy()

# %%
cate = 

# %%
t

# %%
p = draw_pca(a.df)

# %%
save()

# %%
into_neg.loc[into_neg.isnull().any(1)];

# %%
a._categories

# %%
s = pd.DataFrame([[5,np.NaN,7,9,10, np.NAN],
                 [10,np.NaN,np.NaN, 30, np.NAN],
                 [15,30,14,1, 10, np.NAN]]).T
s

# %%
s = s.values

# %%
ss = s[1:5, 1]

# %%
ss

# %%
s[1:5, 1]

# %%
ss[np.isnan(ss)] = 5
s[1:5, 1] = ss
s

# %%
ss = 2
ss

# %%
s.apply(lambda x: x.fillna(x.mean()), axis=1)

# %%
y_true = np.array([[i for j in range(3)] for i in [0.1, 0.15, 0.3, 0.5, .75, 1]]).ravel()

# %%
t = into_neg.iloc[:, 1:].dropna()

pca_df = draw_pca(t / (t.mean(0)/t.mean(0).max()))

# %%
t = into_neg.iloc[:, 1:].dropna()

pca_df = draw_pca(t / (t.mean(0)/t.mean(0).max()))

# %%
pca_df = draw_pca(into_neg.iloc[:, 1:].fillna(10))

# %%
d = {'a':123, 'b': 432}

# %%
d.get('c')


# %%
def draw_pca(df, fi=False, plot_cate=True):
    df_original_cols = df.columns
    qcs, filters, tests = get_index(df.columns)
    if not fi:
        df = df[df.columns^df.columns[filters]]
    hues = []
    for col in df.columns:
        hues.append(classify(col))
        
    cols = df.columns
    data = StandardScaler().fit_transform(df.values.T)
    pca = PCA(n_components=8, whiten=True).fit_transform(data)
    #pca = PCA(n_components=8).fit_transform(df.values.T)
    pca_df = pd.DataFrame(pca)
    pca_df['cate'] = hues
    pca_df.index=cols
    colormap = {'QC':'grey', 'QC_cond':'lightgrey',
                'A': 'coral', 'B': 'orange', 'C': 'forestgreen',
                'D': 'yellow', 'E': 'crimson', 'F': 'royalblue', 'Filter': 'navy', 
                'Filter_050':'#ffff66', 'Filter_015':'#ffffcc', 'Filter_075':'#ffff33', 
                'Filter_030':'#ffff99', 'Filter_010':'#ffffff', 'Filter_100':'#ffff00'}
    fig, ax = plt.subplots(1,1, figsize=(10,6))
    sns.scatterplot(0, 1, hue='cate', data=pca_df, s=300, palette=colormap, alpha=.8, ax=ax)
    
    pca_df = pca_df.reindex(df_original_cols).dropna()
    
    dic_already_plotted = {}
    for i, ind in enumerate(pca_df.index):
        if plot_cate:
            cate_simple = pca_df.loc[ind, 'cate']
            if not dic_already_plotted.get(cate_simple):
                ax.text(pca_df.loc[ind, 0], pca_df.loc[ind, 1], cate_simple, color='maroon', fontsize=14)
                dic_already_plotted[cate_simple] = True
        ax.text(pca_df.loc[ind, 0], pca_df.loc[ind, 1]+.1, i+1, color='k', fontsize=14)
    ax.legend(loc='center left', fontsize=18, bbox_to_anchor=[1,.5])
    plt.tight_layout()
    return pca_df


# %%
((into_neg.iloc[:, 25][:, None]/into_neg.iloc[:, 1:]).median(0)).shape, into_neg.iloc[:, 1:].shape

# %%
t_neg = into_neg.iloc[:, 1:] * ((into_neg.iloc[:, 25][:, None]/into_neg.iloc[:, 1:]).median(0))

# %%
t_neg = into_neg.iloc[:, 1:] * ((into_neg.iloc[:, 25][:, None]/into_neg.iloc[:, 1:]).median(0))

# %%
t_pos = into_pos.iloc[:, 1:] * ((into_pos.iloc[:, 28][:, None]/into_pos.iloc[:, 1:]).median(0))

# %%
pca_df = draw_pca(t_neg.fillna(10))

# %%
pca_df = draw_pca(t_pos.fillna(10))

# %%
pca_df = draw_pca(into_neg.iloc[:, 1:].fillna(10))

# %%
pca_df = draw_pca(into_neg.iloc[:, 1:].fillna(10), fi=True, plot_cate=False)

# %%
pca_df = draw_pca(into_neg.iloc[:, 1:].fillna(10), fi=True)

# %%
fig, [ax1, ax2, ax3] = plt.subplots(3,1,figsize=(16,12))

_ = pca_df[pca_df.index.str.contains('QC')]
ax1.plot(pca_df[0], ':o', c='k', label='Test samples')
ax1.scatter(_.index, _[0], marker='*', s=300, c='tab:red', zorder=10, label='QCs')
ax1.set_xticklabels([])
ax1.set_title('PCA component 1', fontsize=20)
ax1.hlines(_[0].mean(), -10, 100, linestyle='--', color='forestgreen', label='mean')
ax1.set_xlim(-1,46)
ax1.legend(fontsize=15)
ax1.set_xticks([])

ax2.plot(pca_df[1], ':o', c='k', label='Test samples')
ax2.scatter(_.index, _[1], marker='*', s=300, c='tab:red', zorder=10, label='QCs')
ax2.set_xticklabels([])
ax2.set_title('PCA component 2', fontsize=20)
ax2.hlines(_[1].mean(), -10, 100, linestyle='--', color='forestgreen', label='mean')
ax2.set_xlim(-1,46)
ax2.legend(fontsize=15)
ax2.set_xticks([])

ax3.plot(pca_df[2], ':o', c='k', label='Test samples')
ax3.scatter(_.index, _[2], marker='*', s=300, c='tab:red', zorder=10, label='QCs')
ax3.set_xticklabels([])
ax3.set_title('PCA component 3', fontsize=20)
ax3.hlines(_[2].mean(), -10, 100, linestyle='--', color='forestgreen', label='mean')
ax3.set_xlim(-1,46)
ax3.legend(fontsize=15)
ax3.set_xticks([]);

# %%
t = into_pos.iloc[:, 1:].dropna()

pca_df = draw_pca(t / (t.mean(0)/t.mean(0).max()))

# %%
pca_df = draw_pca(into_pos.iloc[:, 1:].fillna(10), fi=True)

# %%
fig, [ax1, ax2, ax3] = plt.subplots(3,1,figsize=(16,12))
ax1.plot(pca_df[0], ':o', c='k', label='Test samples')
_ = pca_df[pca_df.index.str.contains('QC')]
ax1.scatter(_.index, _[0], marker='*', s=300, c='tab:red', zorder=10, label='QCs')
ax1.set_xticklabels([])
ax1.set_title('PCA component 1', fontsize=20)
ax1.hlines(_[0].mean(), -10, 100, linestyle='--', color='forestgreen', label='mean')
ax1.set_xlim(-1,55)
ax1.legend(fontsize=15)
ax1.set_xticks([])

ax2.plot(pca_df[1], ':o', c='k', label='Test samples')
ax2.scatter(_.index, _[1], marker='*', s=300, c='tab:red', zorder=10, label='QCs')
ax2.set_xticklabels([])
ax2.set_title('PCA component 2', fontsize=20)
ax2.hlines(_[1].mean(), -10, 100, linestyle='--', color='forestgreen', label='mean')
ax2.set_xlim(-1,55)
ax2.legend(fontsize=15)
ax2.set_xticks([])

ax3.plot(pca_df[2], ':o', c='k', label='Test samples')
ax3.scatter(_.index, _[2], marker='*', s=300, c='tab:red', zorder=10, label='QCs')
ax3.set_xticklabels([])
ax3.set_title('PCA component 3', fontsize=20)
ax3.hlines(_[2].mean(), -10, 100, linestyle='--', color='forestgreen', label='mean')
ax3.set_xlim(-1,55)
ax3.legend(fontsize=15)
ax3.set_xticks([]);

# %%
test = into_neg.iloc[:, 1:].dropna()
test = pd.DataFrame(test.sum(0))
test['cate'] = test.index
test['cate'] = test.apply(lambda x: classify(x['cate']), axis=1)

# %%
for index in index_neg_qc:
    print((into_neg.iloc[:, 25]/into_neg.iloc[:, index]).median())

# %%
for index in index_pos_qc:
    if index<8:
        print('cond~~~', end='~')
    print((into_pos.iloc[:, 28]/into_pos.iloc[:, index]).median())

# %%
index

# %%
into_pos = into_pos.dropna()

# %%
for index in into_pos.columns[1:]:
    print(f'{index}:{(into_pos.iloc[:, 28]/into_pos[index]).median()}')

# %%
import statsmodels.api as sm

# %%
lowess = sm.nonparametric.lowess(into_pos.iloc[:, index_pos_qc].median(0), index_pos_qc)
plt.figure(figsize=(15,6))
plt.plot(index_pos_qc, into_pos.iloc[:, index_pos_qc].median(0))
plt.plot(lowess[:,0], lowess[:,1])
plt.plot(np.arange(0,54), np.poly1d(np.polyfit(lowess[:,0], lowess[:,1], 5))(np.arange(0, 54)))
plt.tick_params('x', rotation=30)
plt.xticks(index_pos_qc, into_pos.iloc[:, index_pos_qc].columns);

# %%
lowess = sm.nonparametric.lowess(into_neg.iloc[:, index_neg_qc].median(0), index_neg_qc)
plt.figure(figsize=(15,6))
plt.plot(index_neg_qc, into_neg.iloc[:, index_neg_qc].median(0))
plt.plot(lowess[:,0], lowess[:,1])
plt.tick_params('x', rotation=30)
plt.xticks(index_neg_qc, into_neg.iloc[:, index_neg_qc].columns);

# %%
lowess = sm.nonparametric.lowess(into_pos.iloc[:, index_pos_qc].median(0), index_pos_qc)
plt.figure(figsize=(15,6))
plt.plot(index_pos_qc, into_pos.iloc[:, index_pos_qc].median(0))
plt.plot(lowess[:,0], lowess[:,1])
plt.plot(np.arange(0,54), np.poly1d(np.polyfit(lowess[:,0], lowess[:,1], 5))(np.arange(0, 54)))
plt.tick_params('x', rotation=30)
plt.xticks(index_pos_qc, into_pos.iloc[:, index_pos_qc].columns);

# %%
#when dropna, what happened
lowess = sm.nonparametric.lowess(into_neg.iloc[:, index_neg_qc].median(0), index_neg_qc)
plt.figure(figsize=(15,6))
plt.plot(index_neg_qc, into_neg.iloc[:, index_neg_qc].median(0))
plt.plot(lowess[:,0], lowess[:,1])
plt.tick_params('x', rotation=30)
plt.xticks(index_neg_qc, into_neg.iloc[:, index_neg_qc].columns);

# %%
g = sns.catplot(x='cate', y=0, data=test, kind='bar')
fig = g.fig
fig.set_size_inches(15,6)

# %%
plt.figure(figsize=(15,10))
ax1 = plt.axes([0.9, 0.1, 0.1, 0.8]) # up
ax2 = plt.axes([0.1, 0.1, 0.8, 0.8]) # main
ax3 = plt.axes([0.1, 0.9, 0.8, 0.1]) # bottom
ax4 = plt.axes([0.9, 0.9, 0.1, 0.1])
ax1.set_xticks([])
ax1.set_yticks([])
ax3.set_xticks([])
ax3.set_yticks([])
ax4.set_xticks([])
ax4.set_yticks([])
sns.scatterplot('rtmed', 'mzmed', data=fd_pos, alpha=.3, ax=ax2, zorder=10)
sns.kdeplot(fd_pos['rtmed'], shade=True, ax=ax1, vertical=True, legend='')
sns.kdeplot(fd_pos['mzmed'], shade=True, ax=ax3, legend='')

# %%
save()
