---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region {"toc-hr-collapsed": true} -->
# Data import 
<!-- #endregion -->

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

import dill
import time

from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score

from multiprocessing import Pool
import multiprocessing as mp
from functools import partial
plt.style.use('seaborn')
```

```python
dill.load_session('13 09-10Apr.jupyterData')
```

```python
f = pd.read_csv('result_CAMERA.csv').rename({'Unnamed: 0': 'F_index'}, axis=1)
f.head()
```

```python
fd = pd.concat([f.iloc[:, [0]], f.iloc[:, 1:9]], axis=1)
fd.columns
```

```python
into = pd.concat([f.iloc[:, [0]], f.iloc[:, 9:-3]], axis=1)
into.columns
```

```python
fd2 = pd.read_csv('fd_centroid.csv').rename({'Unnamed: 0':'F_index'}, axis=1)
into2 = pd.read_csv('int_centroid.csv').rename({'Unnamed: 0':'F_index'}, axis=1)
into2.head()
```

```python
CAMERA = pd.concat([f.iloc[:, [0]], f.iloc[:, -3:]], axis=1)
CAMERA.columns
```

```python
into.shape
```

## Number of samples were involved in every feature

```python
fd['X1'].value_counts().sort_index(ascending=False)
```

```python
fd.columns
```

## First distribution

```python
plt.style.use('seaborn')
plt.figure(figsize=(15,10))
scatters = plt.scatter(fd['rt'], fd['mz'], alpha=.2)
ax = plt.gca()
ax.set_xlabel('RT(s)', fontsize=15)
ax.set_ylabel('m/z', fontsize=15)
ax.set_title('Distribution of Features', fontsize=20)
#plt.savefig('dist1.png')
```

```python
scatters.get_facecolors()
```

<!-- #region {"toc-hr-collapsed": true} -->
# Filters
<!-- #endregion -->

```python
into.columns
```

```python
filters = into.iloc[:, 0:7]
filters = filters.iloc[:,[0,1,6,2,3,4,5]]
filters = filters.rename({'Filter_01.mzXML':  '0.10',
                          'Fiter_015.mzXML': '0.15',
                          'Filter_03.mzXML':  '0.30', 
                          'Filter_05.mzXML':  '0.50', 
                          'Filter_075.mzXML': '0.75',
                          'Filter_1.mzXML':   '1.00'}, 
                          axis=1)
filters.head()
```

```python
def get_R2(x):
    y_i = np.array([0.1, 0.15, 0.3, 0.5, 0.75, 1])
    y_hat = x[1:] / x[-1]
    return r2_score(y_i, y_hat)
```

```python
mp.cpu_count()
```

## Multiprocess apply

```python
def deco(fun):
    def wrapper(*args, **kwargs):
        s = time.time()
        results = fun(*args, **kwargs)
        print(time.time()-s)
        return results
    return wrapper

def process(data, fun, axis):
    return data.apply(fun, axis)

def best_number_of_processes(data):
    import multiprocessing as mp
    
    CPUs = mp.cpu_count()
    rows = data.shape[0]
    if rows < 10:
        return 2
    elif rows < 100:
        return 4
    else:
        return CPUs
@deco
def multiprocessing_apply(data, fun, processes=None, axis=1):
    from multiprocessing import Pool
    from functools import partial
    
    if not processes:
        processes = best_number_of_processes(data)
        
    with Pool(processes=processes) as pool:
        if data.isnull().sum().sum():
            print("There are NA")
        
        data = np.array_split(data, processes)
        data = pool.map(partial(process, fun=fun, axis=axis), data)
        
        return pd.concat(data)
```

```python
plt.style.use('seaborn')
r2 = multiprocessing_apply(filters.dropna(), get_R2)
sns.distplot(r2[r2>0], color='maroon', bins=50)
plt.xlabel('$R^2$')
```

```python
test = filters.copy()
```

```python
def subbin(nums):
    results = []
    N = len(nums)
    for i in range(2**N):
        pivot = bin(i)[2:][::-1]
        print(pivot)
        re = []
        for j in range(len(pivot)):
            if int(pivot[j]):
                re.append(nums[int(j)])
        results.append(re)
        
    return results
```

```python
def subsets(nums):
    output = [[]]
    for i in range(len(nums)):
        for j in range(len(output)):
            output.append(output[j]+[nums[i]])
            
    return output

f = lambda x: len(x) >= 3

def get_R2_modify(x):
    if x.isnull().sum() > 3:
        return -1

    x = x.fillna(-1)[1:7]
    y_hat = x / x[-1]
    temp = y_hat
    y_hat = list(y_hat)
    
    y_i = [0.1, 0.15, 0.3, 0.5, 0.75, 1]
    pivot = 0
    for i in range(6):
        if x[i] == -1:
            y_hat.pop(i-pivot)
            y_i.pop(i-pivot)
            pivot += 1

    pivots = list(range(len(y_i)))
    subs = subsets(pivots)
    
    subs = list(filter(f, subs))
    y_i = np.array(y_i)
    y_hat = np.array(y_hat)

    d = {}
    for var, conc in zip(temp.values, temp.index):
        d[var] = conc  
    re = float('-inf')
    
    for sub in subs:
        p = r2_score(y_i[sub], y_hat[sub])
        if p > re:
            re = p
            indices = []
            for y in y_hat[sub]:
                indices.append(d[y])         
    return re, indices,        
    #return max(map(lambda index: r2_score(y_i[index], y_hat[index]), subs))
```

```python
r2 = multiprocessing_apply(test, get_R2_modify)
```

```python
r2 = pd.DataFrame([r2.str[0], r2.str[1]]).T
```

```python
r2[0] = r2[0].astype('float')
```

```python
r2[r2[0] > .8][1].value_counts()
```

```python
r2[r2[0] > .8].shape
```

```python
sns.distplot(r2[r2[0] > 0][0], color='maroon')
```

```python
def subsets(nums):
    output = [[]]
    for i in range(len(nums)):
        for j in range(len(output)):
            output.append(output[j]+[nums[i]])
    return output
```

```python
def first_drop(x):
    x = x.fillna(-1)
    x = x[1:7]
    pivot = True
    if x[3] == -1 or x[4] == -1 or x[5] == -1:
        return False
    for i in range(5,-1,-1):
        
        if pivot:
            if x[i] == -1:
                pivot = False
        else:
            if not x[i] == -1:
                return False
    
    return True
```

```python
def get_R2_modify(x):
    x = x.fillna(-1)
    x = x[1:7]

    y_i = np.array([0.1, 0.15, 0.3, 0.5, 0.75, 1])
    
    y_hat = x / x[-1]
    
    pivot = 0
    for i in range(6):
        if not  x[i] == -1:
            pivot = i
            break
           
    y_i = y_i[pivot:]
    y_hat = y_hat[pivot:]
    return r2_score(y_i, y_hat)
```

```python
test2 = filters.copy()
test2['normal'] = test2.apply(first_drop, axis=1)
print(test2.shape)
test2 = test2[test2['normal'] == True]
print(test2.shape)
test2['R2'] = test2.apply(get_R2_modify, axis=1)
test2[test2['R2'] > .8].shape
```

```python
sns.distplot(test2[test2['R2']>0]['R2'])
```

```python
index2 = test2[test2['R2'] > .8].index
```

```python
index1 = r2[r2[0] > .8].index
```

```python
index2.shape, index1.shape
```

```python
(index1 ^ index2).shape
```

```python
(index1 | index2).shape
```

```python
(index1 & index2).shape
```

```python
2589+2472
```

```python
(index2 ^ index1).shape
```

```python
fd['R2'] = r2[0]
```

```python
fd['pass'] = fd['R2'] > .8
```

```python
fd[~fd['pass']]['mz'].shape
```

```python
r2[r2[0] > .8][1].value_counts()
```

```python
t = pd.concat([filters.iloc[:, 1:], r2], axis=1)
t[1] = t.apply(lambda x: str(x[1]), axis=1)
t[t[0]> .8].groupby(1).mean().iloc[:, :-1].plot()
plt.xticks(rotation=30)
```

```python
t[t[0]>.8][1].value_counts()
```

```python
t[t[0]> .8].groupby(1).mean().iloc[:, :-1].astype('int').reindex(t[t[0]>.8][1].value_counts().index).transform(lambda x: np.log(x))
```

```python
t[t[0]> .8].groupby(1).mean().iloc[:, :-1].astype('int').reindex(t[t[0]>.8][1].value_counts().index)
```

```python
sum_t = t[t[0]>.1].iloc[:, :6].sum()
print(sum_t)
sum_t = sum_t / sum_t[-1]
r2_score([.1, .15, .3, .5, .75, 1.], sum_t)
```

```python
t.shape
```

```python
plt.figure(figsize=(15,10))
ax1 = plt.axes([0.9, 0.1, 0.1, .8])
sns.kdeplot(fd[~fd['pass']]['mz'], shade=True, color=[0.29803922, 0.44705882, 0.69019608], vertical=True, ax=ax1)
sns.kdeplot(fd[fd['pass']]['mz'], shade=True, color='darkred', vertical=True, ax=ax1)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_ylabel('')
ax1.legend('')

ax2 = plt.axes([0.1, 0.1, 0.8, 0.8])
sns.scatterplot('rt', 'mz', hue='pass', data=fd, alpha=.3, 
                palette=[[0.29803922, 0.44705882, 0.69019608],'darkred'], 
                edgecolor='w', ax=ax2,
                legend=False)
ax2.set_xlabel('RT(s)', fontsize=22)
ax2.set_ylabel('m/z', fontsize=22)
ax2.tick_params(labelsize=15)

'''top = 900
left = 920
ax2.text(left+40, top, 'Not passed', fontsize=25)
ax2.text(left+40, top-60, 'Passed', fontsize=25)
ax2.scatter(left, top+15, c=[[0.29803922, 0.44705882, 0.69019608]], s=300)
ax2.scatter(left, top-45, c='darkred', s=300)'''

ax3 = plt.axes([0.1, 0.9, 0.8, 0.1])
sns.kdeplot(fd[~fd['pass']]['rt'], shade=True, color=[0.29803922, 0.44705882, 0.69019608], ax=ax3)
sns.kdeplot(fd[fd['pass']]['rt'], shade=True, color='darkred', ax=ax3)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_xlabel('')
ax3.legend('')

ax4 = plt.axes([0.9, 0.9, 0.1, 0.1])
ax4.set_xticks([])
ax4.set_yticks([]) 

#plt.savefig('distribution.png')
```

# Order of Sample Sequence

```python
snames = ['QC 2-3', 'QC 1-1', 'QC 2-2', 'QC 4-2', 'QC 1-3', 'QC 5-2', 'QC 5-1', 'QC 2-1', 
         'Sample 4-2', 'Filter 0.3', 'Sample 1-2', 'QC 3-1',
         'Sample 5-2', 'Sample 3-2', 'Sample 1-3', 'QC 1-2',
         'Sample 5-1', 'Sample 2-2', 'Sample 5-3', 'QC 5-3',
         'Filter 0.75', 'Sample 3-3', 'Sample 4-1', 'QC 3-3',
         'Filter 1.0', 'Sample 2-3', 'Sample 4-3', 'QC 3-2', 
         'Filter 0.1', 'Filter 0.5', 'Sample 1-1', 'QC 4-3', 
         'Sample 2-1', 'Filter 0.15', 'Sample 3-1', 'QC 4-1']
```

```python
snames_df = pd.DataFrame(snames, columns=['Sname'])
```

```python
def classify(x):
    return x[0][:2]
```

```python
snames_df['Class'] = snames_df.apply(classify, axis=1)
snames_df.head()
```

```python
fp1 = FontProperties(fname="../movingdisk/GitHub/python_study/fontawesome-free-5.7.2-desktop/otfs/Font Awesome 5 Free-Solid-900.otf")

cmap = dict(QC='darkblue', Sa='peru', Fi='c')

plt.style.use(plt.style.available[6])

plt.figure(figsize=(16,7.5))

ax=plt.gca()

for i in range(snames_df.shape[0]):
    x = (i % 8)*2.5 + 1.5
    y = -(i // 8)
    cls = snames_df.iloc[i,1]
    sname = snames_df.iloc[i,0]
    ax.text(x, y, '\uf492', fontproperties=fp1, size=50, color=cmap[cls],
           ha='center', va='center')
    ax.text(x, y-0.4, sname, size=20, color=cmap[cls],
           ha='center', va='center')
ax.set_xlim(0,20.5)
ax.set_ylim(-5,0.6)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
#plt.savefig('Injection sequence.png')
```

```python
QCs_eq = into.iloc[:, [12, 7, 11, 17, 9, 20, 19, 10]]
QCs_nor = into.iloc[:, [13, 8, 21, 15, 14, 18, 16]]
```

```python
Samples = into.iloc[:, 22:]
Samples_ordered = Samples.iloc[:, [10, 1, 13, 7, 2, 12, 4, 14, 8, 9, 5, 11, 0, 3, 6]]
```

```python
def RSD(x):
    return int(np.log10(np.mean(x))), np.std(x, ddof=1) / np.mean(x)
```

```python
QCs_nor.dropna(how='all').shape
```

```python
rsd = QCs_nor.dropna(how='all').fillna(1).apply(RSD, axis=1).str[1]
```

```python
sns.distplot(rsd)
```

```python
index2
```

```python
(rsd.index & index2).shape
```

```python
index2.shape
```

```python
rsd.shape
```

```python
cv = rsd.loc[index2&rsd.index]
```

```python
index2.shape
```

```python
cv = pd.cut(cv, bins=(0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, np.inf), labels=('<5', '5-10', '10-15', '15-20', '20-25', '25-30', '>30')).value_counts()
```

```python
cv = cv.sort_index()
```

```python
cv
```

```python
plt.bar(x=range(cv.shape[0]), height=cv)
plt.gca().set_xticklabels(cv.index.values)
```

```python
fd['pass'].value_counts()
```

```python
rsd[rsd<.2].shape
```

```python
from matplotlib_venn import venn2
```

```python
A = set(t[t[0] > .8].index)
B = set(rsd[rsd<.2].index)
venn2([A, B], set_labels=('$R^2>.8$', '$RSD<.2$'))
ax = plt.gca()
```

```python
C = set(test2[test2['R2'] > .8].index)
B = set(rsd[rsd<.2].index)
venn2([C, B], set_labels=('$R^2>.8$', '$RSD<.2$'))
ax = plt.gca()
```

```python
(test2[test2['R2'] > .8].index & rsd[rsd<.2].index).shape
```

```python
(t[t[0] > .8].index & rsd[rsd<.2].index).shape
```

```python
((test2[test2['R2'] > .8].index & rsd[rsd<.2].index) | (t[t[0] > .8].index & rsd[rsd<.2].index)).shape
```

```python
newindex = test2[test2['R2'] > .8].index
newindex.shape
```

```python
test2[test2['R2'] > .8].index.shape
```

```python
t[t[0] > .8].index.shape
```

# Peak Figure 3D

```python
conc = '1.00'
```

```python
threeD = pd.merge(fd, filters, on='F_index').loc[:,['mz', 'rt', conc]].dropna()

mzmin, mzmax = threeD['mz'].min(), threeD['mz'].max()

mzmin = round(mzmin)
mzmax = round(mzmax)

rt_range = np.arange(1, 1501, 1500/500)
mz_range = np.arange(mzmin, mzmax, (mzmax-mzmin)/500)

threeD['RT'] = pd.cut(threeD['rt'], bins=500, labels=rt_range)
threeD['mz'] = pd.cut(threeD['mz'], bins=500, labels=mz_range)

threeD = threeD[[conc, 'RT', 'mz']]

x = rt_range
y = mz_range
x, y = np.meshgrid(x, y)

mat = threeD.groupby(['RT', 'mz']).mean()

mat = mat.unstack().fillna(1).T

z = mat.values.ravel().reshape(500,500)

#z = np.log(z)
```

```python
z.shape
```

```python
plt.contourf(x, y, z, cmap=plt.cm.BuPu)
```

```python
z.min()
```

```python
plt.hist(z[z!=0], bins=20)
```

```python
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 
fig = plt.figure(figsize=(12,10))

ax = Axes3D(fig)

surf = ax.plot_surface(x, y, z, cmap=plt.cm.Greens, 
                       rstride=10, cstride=10, 
                      edgecolor=None, linewidth=0)

fig.colorbar(surf, shrink=.5)

os.system('mkdir peak3D')
for angle in range(0, 360, 2):
    ax.view_init(30, angle)
    plt.savefig('peak3D/peak_%.3d.png' % angle)
os.system('magick -delay 10 -loop 0 peak3D//peak_*.png peak3D.gif')
```

```python
from mpl_toolkits.mplot3d import axes3d, Axes3D
plt.style.use('ggplot')
fig = plt.figure(figsize=(10,6))

ax = Axes3D(fig)

surf = ax.plot_surface(x, y, z, 
                       cmap=plt.cm.BuGn, 
                       rstride=10, cstride=10,
                      edgecolor=None, linewidth=0)
fig.set_facecolor('lightgrey')
fig.colorbar(surf, shrink=.6)
ax.set_facecolor('w')

os.system('mkdir peak3D_v1.1')
for angle in range(0, 360, 2):
    ax.view_init(30, angle)
    ax.set_xlabel('Retention Time', fontsize=20)
    ax.set_ylabel('m/z', fontsize=20)
    ax.set_zlabel('Intensity', fontsize=20)
    
    fig.savefig('peak3D_v1.1/peak_%.3d.png' % angle)

os.system('magick -delay 10 -loop 0 peak3D_v1.1//peak_*.png peak3D_v1.1.gif')
```

```python
dill.dump_session('2_17.pyData')
```

```python
peaktable = pd.read_csv('cp_centroid.csv').rename({'Unnamed: 0':'Peak_index'}, axis=1)

peaktable = peaktable[peaktable['sample'] == 5]

peaktable = peaktable[['mz', 'rt', 'into']]

peaktable.isnull().sum()
```

```python
bins_rt = 1000
bins_mz = 500

rts = np.arange(peaktable['rt'].min(), peaktable['rt'].max(), (peaktable['rt'].max()-peaktable['rt'].min())/bins_rt)
mzs = np.arange(peaktable['mz'].min(), peaktable['mz'].max(), (peaktable['mz'].max()-peaktable['mz'].min())/bins_mz)

peaktable['rt'] = pd.cut(peaktable['rt'], bins_rt)
peaktable['mz'] = pd.cut(peaktable['mz'], bins_mz)
```

```python
def get_med(x):
    return x[0].right, x[1].right
```

```python
peaktable['mz'], peaktable['rt'] = peaktable.apply(get_med, axis=1).str
```

```python
raw_mat = peaktable.groupby(['rt','mz']).mean()
raw_mat = raw_mat.unstack()
raw_mat = raw_mat.fillna(1)
```

```python
raw_mat.shape
```

```python
rt0 = peaktable['rt'].sort_values().unique()
mz0 = peaktable['mz'].sort_values().unique()
rt0, mz0 = np.meshgrid(rt0, mz0)
z0 = raw_mat.values.ravel().reshape(844,452).T
z0 = np.log(z0)
```

```python
rt0.shape, mz0.shape, z0.shape
```

```python
fig = plt.figure(figsize=(12,10))

ax = Axes3D(fig)

surf = ax.plot_surface(rt0, mz0, z0, cmap=plt.cm.Greens, 
                       #rstride=1, cstride=1, 
                      #edgecolor=None, linewidth=0
                      )

fig.colorbar(surf, shrink=.5)
```

# QC


* **DataSets:** QCs_eq, QCs_nor, Samples, Samples_ordered
* **Indexes:**noise_index, pass_index, pass_modified_index
* **Fun:** RSD( return 'mag' and 'RSD')

```python
QCs_nor.columns
```

```python
def RSD_modified(x):
    return int(np.log10(np.mean(x))), np.mean(x), np.std(x, ddof=1) / np.mean(x)   
```

```python
plt.style.use('seaborn')
```

```python
pass_index = test2[test2['R2'] > .8].index
```

```python
QCs_total = QCs_nor.dropna().copy()
```

```python
QCs_pass = QCs_nor.copy().loc[pass_index].dropna()
```

```python
QCs_noise = QCs_nor.copy().loc[QCs_nor.index ^ pass_index].dropna()
```

```python
QCs_noise.shape
```

```python
QCs_pass.shape
```

```python
QCs_total = QCs_nor.dropna()
QCs_pass = QCs_nor.loc[pass_index | pass_modified_index].dropna()
QCs_noise = QCs_nor.loc[noise_index].dropna()
```

```python
QCs_total['mag'], QCs_total['mean'], QCs_total['RSD'] = QCs_total.apply(RSD_modified, axis=1).str
QCs_pass['mag'], QCs_pass['mean'], QCs_pass['RSD'] = QCs_pass.apply(RSD_modified, axis=1).str
QCs_noise['mag'], QCs_noise['mean'], QCs_noise['RSD'] = QCs_noise.apply(RSD_modified, axis=1).str

QCs_total.shape, QCs_pass.shape, QCs_noise.shape
```

```python
sns.distplot(QCs_noise['RSD'], label='noise', hist=False, kde_kws={'shade':'True'})
sns.distplot(QCs_pass['RSD'], label='passed', hist=False, kde_kws={'shade':'True'})
plt.legend(fontsize=20)
ax= plt.gca()
#ax.set_xlim(-.2,1)
```

```python
QCs_noise[QCs_noise['RSD'] < .2].shape[0]
```

```python
QCs_pass[QCs_pass['RSD'] < .2].shape[0]
```

```python
plt.figure(figsize=(15,6))
sns.boxplot('mag', 'RSD', data=QCs_pass, whis=np.inf)
sns.stripplot('mag', 'RSD', data=QCs_pass, jitter=True, edgecolor='k', linewidth=1, alpha=.25, dodge=True)
#plt.savefig('box_strip.png')
```

```python
plt.figure(figsize=(15,6))
sns.boxplot('mag', 'RSD', data=QCs_noise, whis=np.inf)
sns.stripplot('mag', 'RSD', data=QCs_noise, jitter=True, edgecolor='k', linewidth=1, alpha=.25, dodge=True)
#plt.savefig('box_strip_noise.png')
```

```python
plt.style.use('seaborn')
fig = plt.figure(figsize=(16, 12))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

sns.kdeplot(np.log(QCs_pass['mean']), QCs_pass['RSD'], shade=True, cmap=plt.cm.Blues, cut=.5, ax=ax1)
ax1.scatter(np.log(QCs_pass['mean']), QCs_pass['RSD'], alpha=.2, s=10)

sns.kdeplot(np.log(QCs_noise['mean']), QCs_noise['RSD'], shade=True, cmap=plt.cm.Blues, cut=.5, ax=ax2)
ax2.scatter(np.log(QCs_noise['mean']), QCs_noise['RSD'], alpha=.2, s=10)

for i, ax in enumerate([ax1, ax2]):
    ax.set_xticks(np.log([1e5, 1e6, 1e7, 1e8, 1e9]))
    ax.set_xticklabels(['$10^5$', '$10^6$', '$10^7$', '$10^8$', '$10^9$'], fontsize=15)
    ax.set_xlim(11.5, 22)
    ax.set_ylim(0,2)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.set_ylabel('RSD', fontsize=22)
    ax.set_xlabel('')
ax2.set_ylabel('')
ax2.set_yticklabels([])

# ax3
n_peaks_pass = QCs_pass.groupby('mag').aggregate('count').values.T[0] / QCs_pass.shape[0] *100
n_peaks_pass = np.round(n_peaks_pass, 2)
percent_pass = []
for n in n_peaks_pass:
    percent_pass.append('%.2f%%' % n)
bars = ax3.bar(np.arange(6), n_peaks_pass, alpha=1)
c = bars[0].get_facecolor()
for i in range(6):
    ax3.text(i, n_peaks_pass[i]+1.5, percent_pass[i], color=c[:3], ha='center', va='center', fontsize=15)
    
pass_02 = []
for i in range(6):
    per = QCs_pass[(QCs_pass['RSD'] <.2) & (QCs_pass['mag'] == i+4)].shape[0]
    per = per / QCs_pass.groupby('mag').aggregate('count').values.T[0][i] * 100
    pass_02.append(per)
    
ax3.plot(np.arange(6), pass_02, c='maroon', marker='o')

pass_02_str = []
for per in pass_02:
    pass_02_str.append('%.2f%%' % per)

loc_off = [[0,10], [-.5,0], [-.4,3], [-.3,4], [.5,-2], [-.5,0]]
for i in range(6):
    ax3.text(i+loc_off[i][0], pass_02[i]+loc_off[i][1], pass_02_str[i],
            va='center', ha='center',
            fontsize=15, color='maroon')
ax3.set_xticks(np.arange(6))
ax3.set_xticklabels(['$10^4$', '$10^5$', '$10^6$', '$10^7$', '$10^8$', '$10^9$'], fontsize=15)
ax3.set_ylim(0, 105)

# ax4
n_peaks_noise = QCs_noise.groupby('mag').aggregate('count').values.T[0] / QCs_noise.shape[0] *100
n_peaks_noise = np.round(n_peaks_noise, 2)
percent_noise = []
length = len(n_peaks_noise)

for n in n_peaks_noise:
    percent_noise.append('%.2f%%' % n)
    
bars = plt.bar(np.arange(length), n_peaks_noise, alpha=1)
c = bars[0].get_facecolor()
for i in range(length):
    plt.text(i, n_peaks_noise[i]+1.5, percent_noise[i], color=c[:3], ha='center', va='center', fontsize=15)
    
noise_02 = []
for i in range(length):
    per = QCs_noise[(QCs_noise['RSD'] <.2) & (QCs_noise['mag'] == i+5)].shape[0]
    per = per / QCs_noise.groupby('mag').aggregate('count').values.T[0][i] * 100
    noise_02.append(per)
    
plt.plot(np.arange(length), noise_02, c='maroon', marker='o')

noise_02_str = []
for per in noise_02:
    noise_02_str.append('%.2f%%' % per)

loc_off = [[-.2,-4], [.5,-1], [.4,2], [.5,3], [.5,2], [.5,0], [0.5,4]]
for i in range(length):
    plt.text(i+loc_off[i][0], noise_02[i]+loc_off[i][1], noise_02_str[i],
            va='center', ha='center',
            fontsize=15, color='maroon')

ax4.set_xticks(np.arange(length))
ax4.set_xticklabels(['$10^5$', '$10^6$', '$10^7$', '$10^8$', '$10^9$','$10^{10}$'], fontsize=15)

for i, ax in enumerate([ax3, ax4]):
    ax.set_ylabel('% peaks', fontsize=22)
    ax.yaxis.set_tick_params(labelsize=15)
    labels = ['(Passed)', '(Not Passed)']
    ax.set_xlabel('Intensities\n%s' % labels[i], fontsize=22)
    

rect = Rectangle((4.5, 85), 1, 2.5)
ax4.add_patch(rect)
ax4.text(6, 86, '%peaks', ha='center', va='center', fontsize=15)
line = Line2D([4.5, 5.5], [77, 77], color='maroon', marker='o')
ax4.text(6.1, 76, 'RSD \nwithin 20%', ha='center', va='center', fontsize=15)
ax4.add_line(line)
ax4.set_ylabel('')
ax4.set_yticklabels([])
ax4.set_ylim(0,105)

plt.tight_layout()
#plt.savefig('compare.png')
```

```python
QCs_pass[QCs_pass['RSD'] >.2].shape
```

```python
QCs_final = QCs_pass[QCs_pass['RSD'] <.2]
QCs_final = QCs_eq.loc[QCs_final.index].join(QCs_final).dropna().iloc[:, :15]
```

```python
QCs_final.shape
```

```python
QCs_final.columns
```

```python
from sklearn.decomposition import PCA
```

```python
QCs_PCA = PCA(n_components=8).fit_transform(np.log(QCs_final.T))
```

```python
QCs_PCA.shape
```

```python
Samples_final = Samples.iloc[QCs_final.index]
```

```python
allsamples = QCs_final.join(Samples_final).dropna()

seq = [0, 1, 2, 3, 4, 5, 6, 7, 
      25, 16, 8, 
      28, 22, 17, 9,
      27, 19, 29, 10, 
      23, 24, 11, 
      20, 26, 12, 
      15, 13, 
      18, 21, 14]

allsamples = allsamples.iloc[:,seq]
```

```python
allsamples.columns
```

```python
PCA_allsamples = PCA(n_components=8).fit_transform(np.log(allsamples.T))
```

```python
def deal_names(cols):
    re = []
    for i, name in enumerate(cols):
        if i in [0, 1, 2]:
            re.append('QC-eq-%s' % str(i+1))
        elif i in [3, 4, 5, 6, 7]:
            re.append('QC-eq')
        elif name.startswith('Sa'):
            name = name[:10]
            re.append('Sa-%s' % name[-3])
        else:
            re.append('QC-nor')
    
    return re
```

```python
names = deal_names(allsamples.columns)
```

```python
colormap = {'QC-eq-1': ['darkred',True], 'QC-eq-2':['red', True], 'QC-eq-3':['lightcoral', True], 
            'QC-eq': ['gold', True], 'QC-nor': ['olive', True], 
           'Sa-1': ['plum', True], 'Sa-2': ['skyblue', True], 'Sa-3': ['teal', True], 'Sa-4': ['forestgreen', True], 'Sa-5': ['peru', True]}
colormap
```

```python
pca_mat = pd.DataFrame(PCA_allsamples)

pca_mat['class'] = names
pca_mat['cate'] = pca_mat['class'].str[:2]
pca_mat.head()
```

```python
plt.figure(figsize=(12,8))
scatters = sns.scatterplot(0, 1, hue='class', style='cate', data=pca_mat, s=200)
plt.legend(fontsize=18, bbox_to_anchor=[1,1, .2, 0])
plt.xlabel('Component 1', fontsize=18)
plt.ylabel('Component 2', fontsize=18)

#plt.savefig('pca_all.png')
```

```python
def deal_names(cols):
    re = []
    for i, name in enumerate(cols):
        if i in [0, 1, 2]:
            re.append('QC-eq-%s' % str(i+1))
        elif i in [3, 4, 5, 6, 7]:
            re.append('QC-eq')
        elif name.startswith('Sa'):
            name = name[:10]
            re.append('Sa-%s' % name[-3])
        else:
            re.append('QC-nor')
    
    return re

def draw_pca(PCA_allsamples):
    names = deal_names(allsamples.columns)
    colormap = {'QC-eq-1': ['darkred',True], 'QC-eq-2':['red', True], 'QC-eq-3':['lightcoral', True], 
            'QC-eq': ['gold', True], 'QC-nor': ['olive', True], 
           'Sa-1': ['plum', True], 'Sa-2': ['skyblue', True], 'Sa-3': ['teal', True], 'Sa-4': ['forestgreen', True], 'Sa-5': ['peru', True]}
    pca_mat = pd.DataFrame(PCA_allsamples)

    pca_mat['class'] = names
    pca_mat['cate'] = pca_mat['class'].str[:2]
    
    plt.figure(figsize=(12,8))
    scatters = sns.scatterplot(0, 1, hue='class', style='cate', data=pca_mat, s=200)
    plt.legend(fontsize=18, bbox_to_anchor=[1,1, .2, 0])
    plt.xlabel('Component 1', fontsize=18)
    plt.ylabel('Component 2', fontsize=18) 
```

```python
draw_pca(PCA().fit_transform(allsamples.T))
```

```python
fig = plt.figure(figsize=(16,12))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

ax1.plot(pca_mat[0], ':o', c='k', label='Samples')
ax1.scatter(pca_mat[pca_mat['cate'] == 'QC'].index, pca_mat[pca_mat['cate'] == 'QC'][0], marker='*', s=300, c='tab:red', zorder=10, label='QCs')
ax1.set_xticklabels([])
ax1.set_title('PCA component 1', fontsize=20)
ax1.hlines(pca_mat[pca_mat['cate'] == 'QC'][0].mean(), -10, 40, linestyle='--', color='forestgreen', label='mean')
ax1.set_xlim(-1,32)
ax1.legend(fontsize=15)

ax2.plot(pca_mat[1], ':o', c='k', )
ax2.scatter(pca_mat[pca_mat['cate'] == 'QC'].index, pca_mat[pca_mat['cate'] == 'QC'][1], marker='*', s=300, c='tab:red', zorder=10)
ax2.set_xticklabels([])
ax2.set_title('PCA component 2', fontsize=20)
ax2.hlines(pca_mat[pca_mat['cate'] == 'QC'][1].mean(), -10, 40, linestyle='--', color='forestgreen')
ax2.set_xlim(-1,32)

ax3.plot(pca_mat[2], ':o', c='k', )
ax3.scatter(pca_mat[pca_mat['cate'] == 'QC'].index, pca_mat[pca_mat['cate'] == 'QC'][2], marker='*', s=300, c='tab:red', zorder=10)
ax3.set_title('PCA component 3', fontsize=20)
ax3.hlines(pca_mat[pca_mat['cate'] == 'QC'][2].mean(), -10, 40, linestyle='--', color='forestgreen')
ax3.set_xlim(-1,32)
#plt.savefig('PCA_123.png')
```

```python
lm_test = test.sort_values(by='R2_modified', ascending=False).iloc[[2]][['0.10', '0.15', '0.30', '0.50', '0.75', '1.00']].T
```

```python
lm_test['x'] = [0.1, 0.15, 0.3, 0.5, 0.75, 1]
```

```python
lm_test.rename({lm_test.columns[0]:'y'}, axis=1)
```

```python
test.sort_values(by='R2_modified', ascending=False).iloc[[2000]]
```

```python
sns.lmplot('x', 'y', data=lm_test)
plt.yticks([])
plt.xticks([])
plt.xlabel('')
plt.ylabel('')
plt.savefig('filter_illus.png')
```

```python
def get_R2_final(x):
    x = x.fillna(-1)

    y_i = np.array([0.1, 0.15, 0.3, 0.5, 0.75, 1])
    
    y_hat = x / x[-1]
    
    pivot = 0
    for i in range(6):
        if not  x[i] == -1:
            pivot = i
            break
           
    y_i = y_i[pivot:]
    y_hat = y_hat[pivot:]
    return r2_score(y_i, y_hat)
```

```python
fig = plt.figure(figsize=(16,12))
axs = []
df = test.sort_values(by='R2_modified', ascending=False)[:2000].sample(12)
indices = df.index[:12]
for i in range(12):
    axs.append(fig.add_subplot(3, 4, i+1))

for i, ax in enumerate(axs):
    lm_test = df.loc[[indices[i]]][['0.10', '0.15', '0.30', '0.50', '0.75', '1.00']].T
    lm_test['x'] = [0.1, 0.15, 0.3, 0.5, 0.75, 1]
    lm_test = lm_test.rename({lm_test.columns[0]:'y'}, axis=1)
    lm_test['y'] = lm_test['y'] / lm_test['y'][-1]
    
    sns.regplot('x', 'y', data=lm_test, ax=ax)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim(-.05, 1.05)
    ax.set_ylim(-.05, 1.05)
    
    R2 = get_R2_final(lm_test['y'])
    R2 = round(R2, 5)
    
    index = indices[i]
    
    ax.text(.2, .8, '$R^2$: %s\nNo.%s' % (R2, index), fontsize=20)
    
plt.tight_layout()
plt.savefig('12lmplot.png', dpi=96)
```

```python
xx = np.linspace(0, 25, 1000)
yy = np.zeros(1000)
```

```python
for i, x in enumerate(xx):
    if x < 2:
        yy[i] = 2
    if 2 <= x < 8:
        yy[i] = 2 + (x-2) * 8
    if 8 <= x < 16:
        yy[i] = 50
    if 16 <= x < 17:
        yy[i] = 50 + (x-16) * 48
    if 17 <= x < 21:
        yy[i] = 98
    if 21 <= x < 21.1:
        yy[i] = 98 - (x-21) * 960
    if x >= 21.1:
        yy[i] = 2
```

```python
plt.figure(figsize=(16, 4))
plt.plot(xx, yy, c='darkblue')
plt.fill_between(xx, 0, yy, facecolor='forestgreen', alpha=.5, label='organic phase')
plt.fill_between(xx, yy, 100, facecolor='beige', alpha=1, label='inorganic phase')
plt.xlim(0,25)
plt.ylim(0,100)
plt.legend()
plt.xlabel('Time (min)')
plt.ylabel('%')
plt.tight_layout()
plt.savefig('gradient.png')
```

```python
now = time.strftime('%H %M-%d%h')
dill.dump_session('%s.jupyterData' % now)
```

<!-- #region {"toc-hr-collapsed": false} -->
# PCA
<!-- #endregion -->

```python
def deal_names(cols):
    re = []
    for i, name in enumerate(cols):
        if i in [0, 1, 2]:
            re.append('QC-eq-%s' % str(i+1))
        elif i in [3, 4, 5, 6, 7]:
            re.append('QC-eq')
        elif name.startswith('Sa'):
            name = name[:10]
            re.append('Sa-%s' % name[-3])
        else:
            re.append('QC-nor')
    
    return re

def draw_pca(mat):
    PCA_allsamples = PCA(whiten=False).fit_transform(mat)
    names = deal_names(allsamples.columns)
    colormap = {'QC-eq-1': ['darkred',True], 'QC-eq-2':['red', True], 'QC-eq-3':['lightcoral', True], 
            'QC-eq': ['gold', True], 'QC-nor': ['olive', True], 
           'Sa-1': ['plum', True], 'Sa-2': ['skyblue', True], 'Sa-3': ['teal', True], 'Sa-4': ['forestgreen', True], 'Sa-5': ['peru', True]}
    pca_mat = pd.DataFrame(PCA_allsamples)

    pca_mat['class'] = names
    pca_mat['cate'] = pca_mat['class'].str[:2]
    
    fig, axs = plt.subplots(1,2, figsize=(16,6))
    
    ax = axs[0]
    scatters = sns.scatterplot(0, 1, hue='class', style='cate', data=pca_mat, s=200, ax=ax)
    ax.legend(fontsize=18, bbox_to_anchor=[1,1, .2, 0])
    ax.set_xlabel('Component 1', fontsize=18)
    ax.set_ylabel('Component 2', fontsize=18) 
    
    ax = axs[1]
    sns.scatterplot(0, 1, hue='class', style='class', data=pca_mat[pca_mat['cate']=='QC'], s=200, ax=ax)
    ax.legend(fontsize=18)
    ax.set_xlabel('Component 1', fontsize=18)
    ax.set_ylabel('Component 2', fontsize=18)
    plt.tight_layout()
    return pca_mat

def draw_trend(pca_mat):
    fig = plt.figure(figsize=(16,12))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    ax1.plot(pca_mat[0], ':o', c='k', label='Samples')
    ax1.scatter(pca_mat[pca_mat['cate'] == 'QC'].index, pca_mat[pca_mat['cate'] == 'QC'][0], marker='*', s=300, c='tab:red', zorder=10, label='QCs')
    ax1.set_xticklabels([])
    ax1.set_title('PCA component 1', fontsize=20)
    ax1.hlines(pca_mat[pca_mat['cate'] == 'QC'][0].mean(), -10, 40, linestyle='--', color='forestgreen', label='mean')
    ax1.set_xlim(-1,32)
    ax1.legend(fontsize=15)

    ax2.plot(pca_mat[1], ':o', c='k', )
    ax2.scatter(pca_mat[pca_mat['cate'] == 'QC'].index, pca_mat[pca_mat['cate'] == 'QC'][1], marker='*', s=300, c='tab:red', zorder=10)
    ax2.set_xticklabels([])
    ax2.set_title('PCA component 2', fontsize=20)
    ax2.hlines(pca_mat[pca_mat['cate'] == 'QC'][1].mean(), -10, 40, linestyle='--', color='forestgreen')
    ax2.set_xlim(-1,32)

    ax3.plot(pca_mat[2], ':o', c='k', )
    ax3.scatter(pca_mat[pca_mat['cate'] == 'QC'].index, pca_mat[pca_mat['cate'] == 'QC'][2], marker='*', s=300, c='tab:red', zorder=10)
    ax3.set_title('PCA component 3', fontsize=20)
    ax3.hlines(pca_mat[pca_mat['cate'] == 'QC'][2].mean(), -10, 40, linestyle='--', color='forestgreen')
    ax3.set_xlim(-1,32)
```

```python
pca_mat = draw_pca(allsamples.T)
```

```python
draw_trend(pca_mat)
```

```python
pca_mat = draw_pca(np.log(allsamples.T))
```

```python
draw_trend(pca_mat)
```

```python
f = allsamples.T.values
```

```python
f
```

## 平衡conc, dropna

```python
f1 = f / (f.sum(1) / f.sum(1).max())[:, np.newaxis]
```

```python
f1.shape
```

### 只平衡

```python
f11 = f1
```

```python
pca_mat = draw_pca(f11)
```

```python
draw_trend(pca_mat)
```

### 先平衡再log

```python
f12 = np.log(f1)
```

```python
pca_mat = draw_pca(np.log(f12))
```

```python
draw_trend(pca_mat)
```

### 先log再平衡

```python
f13 = np.log(f)
```

```python
f13 = f13 / (f13.sum(1) / f13.sum(1).max())[:, np.newaxis]
```

```python
pca_mat = draw_pca(f13)
```

```python
draw_trend(pca_mat)
```

## 平衡feature intensity, fropna


### 先平衡feature

```python
sum_median = np.median(f.max(0))
```

```python
np.median(f.sum(0))
```

```python
factors = f.sum(0) / sum_median
```

```python
factors.shape
```

```python
f21 = (f / factors)
```

```python
plt.plot(f.sum(0))
```

```python
pca_mat = draw_pca(f21)
```

```python
draw_trend(pca_mat)
```

### 平衡后再log

```python
f22 = np.log(f21)
```

```python
pca_mat = draw_pca(f22)
```

```python
draw_trend(pca_mat)
```

### 先log 后平衡feature

```python
f.shape
```

## StandardScaler

```python
from sklearn.preprocessing import StandardScaler
```

```python
f0 = StandardScaler().fit_transform(f)
```

### 只StandardScaler

```python
pca_mat = draw_pca(f0)
```

```python
draw_trend(pca_mat)
```

### log后 StandardScaler

```python
f01 = StandardScaler().fit_transform(np.log(f))
```

```python
pca_mat = draw_pca(f01)
```

```python
draw_trend(pca_mat)
```

### 平衡conc后 StandardScaler(没有log)
**也许是最好的**

```python
f3 = f / (f.sum(1) / f.sum(1).max())[:, np.newaxis]
```

```python
f3 = f / (f.sum(1) / f.sum(1).max())[:, np.newaxis]
```

```python
f33 = StandardScaler().fit_transform(f3)
```

```python
pca_mat = draw_pca(f33);
plt.savefig('best_pca.png')
```

```python
draw_trend(pca_mat)
plt.savefig('best_trend.png')
```

### 平衡conc后 log-> StandardScaler

```python
f34 = StandardScaler().fit_transform(np.log(f3))
```

```python
pca_mat = draw_pca(f34)
```

```python
def draw_all(df):
    f = df.T.values
    f = f / (f.sum(1) / f.sum(1).max())[:, np.newaxis]
    f = StandardScaler().fit_transform(f)
    
    pca_mat = draw_pca(f)
    draw_trend(pca_mat)
```

<!-- #region {"toc-hr-collapsed": true} -->
# 测试不同的pass集合
<!-- #endregion -->

## 1361 dropna

```python
draw_all(allsamples)
```

```python
i5000 = t[t[0] > .8].index
i2000 = test2[test2['R2'] > .8].index
```

```python
ints = into.iloc[:, 1:]
```

```python
ints = ints.loc[:, allsamples.columns]
```

```python
ints.loc[i2000]
```

```python
QCs_nor.columns
```

```python
QCs_nor.dropna(how='all').shape
```

```python
index = allsamples.index
qc = QCs_nor.copy().dropna().T.values
pivot = QCs_nor.copy().loc[index].T.values
```

```python
qc = qc / (pivot.sum(1) / pivot.sum(1).max())[:, np.newaxis]
```

```python
qc = pd.DataFrame(qc, index=QCs_nor.dropna().columns, columns=QCs_nor.dropna().index).T
```

```python
qc['mag'], qc['mean'], qc['RSD'] = qc.apply(RSD_modified, axis=1).str
```

```python
QCs_pass[QCs_pass['RSD']<.2].shape
```

```python
allsamples.index
```

```python
QCs_pass.head()
```

```python
_ = qc.loc[qc.index & i2000]
```

```python
_[_['RSD'] < .2].shape
```

```python
_.shape
```

```python
8591 in _.index
```

```python
_.shape
```

## 1426 dropna

```python
draw_all(ints.loc[_[_['RSD'] < .2].index].dropna())
```

```python
draw_all(ints.loc[_[_['RSD'] < .2].index].fillna(0))
```

```python
f_t = pd.DataFrame(f3, index=allsamples.columns, columns=allsamples.index).T
```

```python
f_t.columns = names
```

```python
f_t.columns
```

```python
s1 = f_t['Sa-1']
s2 = f_t['Sa-2']
s3 = f_t['Sa-3']
s4 = f_t['Sa-4']
s5 = f_t['Sa-5']
```

# T test


## s1 and s2

```python
import statsmodels.api as sm
```

```python
a, b = pd.concat([s1.T, s2.T]), pd.concat([s3.T, s4.T, s5.T])
```

```python
a, b = s1.T, s2.T
```

```python
ttest = sm.stats.ttest_ind(a, b)

pvalues_fdr_bh = sm.stats.multipletests(ttest[1], alpha=.05, method='fdr_bh')[1]
pvalues_fdr_bh

pvalues = -np.log10(pvalues_fdr_bh)

times = np.log2(a.mean(0) / b.mean(0)).values

ttest[1][ttest[1] < .05].shape, pvalues_fdr_bh[pvalues_fdr_bh < .05].shape
```

```python
volcano = pd.DataFrame([pvalues_fdr_bh, a.mean(0)/b.mean(0)]).T
volcano.index = a.columns
volcano.head()
```

```python
times.shape, pvalues.shape
```

```python
plt.figure(figsize=(12,8))
volcano_plot(times, pvalues, zorder=10)
fc = 2
plt.fill_between(np.linspace(-100, -np.log2(fc)), np.ones([50])*-np.log10(5e-2), np.ones([50])*-np.log10(5e-6), facecolor='coral', alpha=.5, zorder=1)
plt.fill_between(np.linspace(np.log2(fc), 100), np.ones([50])*-np.log10(5e-2), np.ones([50])*-np.log10(5e-6), facecolor='coral', alpha=.5, zorder=1)
plt.xlim(-7,7)
plt.ylim(-.5, 5.2)
plt.text(-6.5, 2, 'p<0.05\nfold change > %s'% fc, fontsize=16, ha='left', va='center', color='r')
plt.text(3.7, 2, 'p<0.05\nfold change > %s' % fc, fontsize=16, ha='left', va='center', color='r')
plt.xlabel('$log_{2}(Fold Change)$', fontsize=15)
plt.ylabel('$-log_{10}(p-value)$', fontsize=15)
plt.savefig('volcano2.png')
```

```python
def volcano_plot(times, pvalues, **kwargs):
    sns.scatterplot(times, pvalues, **kwargs)
```

```python
plt.style.use('seaborn')
```

```python
plt.figure(figsize=(10,8))
volcano_plot(times, pvalues)
```

```python
ttest[1].argsort()
```

```python
s1.iloc[273]
```

```python
s2.iloc[273]
```

```python
df = pd.read_csv('OPLS.csv', index_col=0)
```

## PLS

```python
from sklearn.cross_decomposition import PLSRegression
```

```python
pls = PLSRegression(n_components=2)
res = pls.fit(pd.concat([s1.T, s2.T, s3.T, s4.T, s5.T]), [0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1])
```

```python
pls_mat = pd.DataFrame(res.x_scores_)

pls_mat['c'] = [0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1]
```

```python
plt.figure(figsize=(8,6))
sns.scatterplot(0, 1, data=pls_mat, hue='c', s=200, legend=False)
plt.savefig('pls.png')
```

```python
VIPs = pd.read_csv('vip_from_pls_da.csv', index_col=0)

VIPs[(VIPs['comp 1'] > 1)&(VIPs['comp 2'] > 1)]
```

## RandomForest

```python
from sklearn.ensemble import RandomForestClassifier
```

```python
rf = RandomForestClassifier(n_estimators=20000, n_jobs=-1, max_depth=1)
```

```python
rf.fit(df.iloc[:, :-1].values, df.iloc[:, -1])
```

```python
fi = rf.feature_importances_
```

```python
fi = pd.DataFrame(fi).rename({0: 'var'}, axis=1)
```

```python
fi['c'] = pd.cut(fi['var'], 10)
fi['c'] = np.where(fi['c'] == pd.cut(fi['var'], 10).value_counts().index[0], 0, 1)
```

```python
pd.cut(fi['var'], 10).value_counts()
```

```python
plt.figure(figsize=(25,5))
sns.scatterplot(x=range(fi.shape[0]), y='var', data=fi, hue='c', size='var')
#plt.savefig('feature_importance.png')
```

```python
fi.index = df.columns[:-1]
```

```python
fd_final = fd.loc[allsamples.index]
```

```python
sns.scatterplot(x='rt', y='mz', data=fd_final)
```

```python
fd_final = fd_final.join(fi, on=fi.index)
```

```python
fig = plt.figure(figsize=(12,6))
sns.scatterplot(x='rt', y='mz', data=fd_final[fd_final['c']==0], alpha=.2, )
sns.scatterplot(x='rt', y='mz', data=fd_final[fd_final['c']==1], size='var')
plt.xlim(0, 1000)
plt.ylim(50, 1000)
```

```python
fig = plt.figure(figsize=(12,6))
sns.scatterplot(x='rt', y='mz', data=fd_final[fd_final['c']==0], alpha=.2, )
sns.scatterplot(x='rt', y='mz', data=fd_final[fd_final['c']==1], size='var')
plt.xlim(0, 1000)
plt.ylim(50, 1000)
plt.savefig('scatter_rf.png')
```

```python
sns.kdeplot(fi['var'], shade=True)
```

# Isotopes, Adducts and Neutral loss

<!-- #region {"toc-hr-collapsed": false} -->
## Find isotopes
<!-- #endregion -->

```python
mzs = fd_final['mz'].values
```

```python
mat = pd.DataFrame(mzs[:, None] - mzs)
```

```python
def iso(x):
    x = x.values[:x.name][::-1]
    l = len(x)
    res = {}
    Da = 1.0034
    
    for i, x0 in enumerate(x):
        i = l-i-1
        if x0 > 5.1:
            break
        n = round(x0)
        sub = np.abs(Da*n - x0)
        if sub < 0.01 and round(x0)>0:
            res[i] = int(round(x0))
            
    if res:
        return res
```

```python
def match_rt(x):
    d = x[0]    
    for x0 in list(d):
        sub = fd_final.iloc[x0, 4] - fd_final.iloc[x.name, 4]
        if np.abs(sub) > .5:
            d.pop(x0)
    if d:
        return d
```

```python
iso_res = pd.DataFrame(mat.apply(iso, axis=1)).dropna()

iso_res = iso_res.apply(match_rt, axis=1).dropna()

iso_res.shape
```

```python
def flatten(l):
    res = []
    for l0 in l:
        if isinstance(l0, list):
            res.extend(flatten(l0))
        else:
            res.append(l0)
            
    return res
```

```python
def info(index, msg=True):
    vice_list= flatten([list(item) for item in iso_res.values])
    if index in iso_res.index:
        pass
    elif index in vice_list:
        index = iso_res[pd.notna(iso_res.str.get(index))].index.values[0]
        print('Index in vica list')
    elif not index in iso_res.index:
        raise IndexError('No isotopes found for index: %s' % index)

    indices = [index]
    keys = list(iso_res.loc[index])
    indices.extend(keys)
    #F_index = fd_final.iloc[index, 0]
    
    for key in keys:
        #host = fd_final.iloc[key, 0]
        var = mat.iloc[index, key]
        var = var - round(var) * 0.0034
        if msg:
            print('ParentIndex->{0}, HostIndex->{1}, Sub<{2:.5f}>'.format(index, indices[1:], var))
            #print('ParentIndex->{0}, HostIndex->{1}, Sub<{2:.5f}>'.format(F_index, host, var))
    res = fd_final.iloc[indices, [1, 4, 11]]
    ints = pd.DataFrame(QCs_nor.iloc[res.index].mean(1)).rename({0:'Intensities in QCs'}, axis=1)
    return res.join(ints)
```

```python
def info(index, msg=True):
    vice_list= flatten([list(item) for item in iso_res.values])
    if index in iso_res.index:
        pass
    elif index in vice_list:
        index = iso_res[pd.notna(iso_res.str.get(index))].index.values[0]
        print('Index in vica list')
    else:
        raise IndexError('No isotopes found for index: %s' % index)

    indices = [index]
    keys = list(iso_res.loc[index])
    indices.extend(keys)
    #F_index = fd_final.iloc[index, 0]
    vals = []
    for key in keys:
        #host = fd_final.iloc[key, 0]
        var = mat.iloc[index, key]
        var = var - round(var) * 0.0034
        vals.append(var)
        if msg:
            print('ParentIndex->{0}, HostIndex->{1}, Sub<{2:.5f}>'.format(index, indices[1:], var))
            #print('ParentIndex->{0}, HostIndex->{1}, Sub<{2:.5f}>'.format(F_index, host, var))
    res = fd_final.iloc[indices, [1, 4, 11]]
    ints = pd.DataFrame(QCs_nor.iloc[res.index].mean(1)).rename({0:'Intensities in QCs'}, axis=1)
    return res.join(ints), vals[0]
```

```python
info(1300)
```

```python
def make_iso_table(data):
    res = []
    for i, d in data.iteritems():
        tb = info(i, msg=False)
        
        tb['fd_index'] = tb.index
        i = [i]
        i.extend(list(d))
        tb['iso_index'] = i
        tb['group'] = i[0]
        res.append(tb)
    return pd.concat(res).drop_duplicates()
```

```python
iso_table = make_iso_table(iso_res)
```

```python
l = flatten([list(item) for item in iso_res.values])

iso_res.index & np.array(l)
```

```python
iso_res.index
```

```python
plt.figure(figsize=(12, 8))
for i in iso_res.index:
    if np.random.random() > .65:
        r = info(i, msg=False)
        sns.scatterplot('rt', 'mz', data=r, s=100, alpha=.7)
```

```python
plt.figure(figsize=(12, 8))
for i in iso_res.index:
    if np.random.random() > .65:
        r = info(i, msg=False)
        sns.scatterplot('rt', 'mz', data=r, s=100, alpha=.7)
plt.savefig('isotope_pairs.png')
```

```python
info(37)
```

```python
def get_sub(x):
    return info(int(x['group']), msg=False)[1]
```

```python
iso_table['sub'] = iso_table.apply(get_sub, axis=1)
```

```python
iso_table['sub_normalized'] = iso_table['sub'] / np.round(iso_table['sub'])
```

```python
iso_table['sub_group'] = pd.cut(iso_table['sub_normalized'], bins=(0.991, 0.999, 1.001, 1.010), labels=('<0.999', '0.999-1.001', '>1.001'))
```

```python
plt.figure(figsize=(20,5))
#plt.scatter(iso_table['rt'], iso_table['sub_normalized'], marker='s', facecolor='w', edgecolor='r', linewidth=1, alpha=.7)
sns.scatterplot(x='rt', y='sub_normalized', data=iso_table, hue='sub_group', s=80)
plt.savefig('isotope_scatter.png')
```

## Adducts

```python
def deco(fun):
    def wrapper(*args, **kwargs):
        s = time.time()
        results = fun(*args, **kwargs)
        print(time.time()-s)
        return results
    return wrapper

def process(data, fun, axis, **kwargs):
    return data.apply(fun, axis, **kwargs)

def best_number_of_processes(data):
    import multiprocessing as mp
    
    CPUs = mp.cpu_count()
    rows = data.shape[0]
    if rows < 10:
        return 2
    elif rows < 100:
        return 4
    else:
        return CPUs
@deco
def multiprocessing_apply(data, fun, processes=None, axis=1, **kwargs):
    from multiprocessing import Pool
    from functools import partial
    
    if not processes:
        processes = best_number_of_processes(data)
        
    with Pool(processes=processes) as pool:
        if data.isnull().sum().sum():
            print("There are NA")
        
        data = np.array_split(data, processes)
        data = pool.map(partial(process, fun=fun, axis=axis, **kwargs), data)
        
        return pd.concat(data)
```

```python
adducts = pd.read_html('adducts.html')[0]

adducts = adducts.dropna().drop([0, 3]).drop([3,5,6], axis=1)

pos_adds = adducts.loc[:39]
neg_adds = adducts.loc[40:]

pos_adds.shape, neg_adds.shape
```

```python
rts = fd_final['rt'].values
```

```python
def find_close_rt(x, rt):
    if np.abs(x['rt'] - rt) < .5:
        return x

def partial_find_close_rt(rt):
    return fd_final.apply(find_close_rt, rt=rt, axis=1, result_type='broadcast').dropna()

@deco
def multi_find_close_rt(rts):
    with Pool(processes=12) as pool:
        res = pool.map(partial_find_close_rt, rts)
        res = list(res)
    return res
```

```python
res_after_find_rt = multi_find_close_rt(rts)

sum([len(r) for r in res_after_find_rt])
```

```python
def parse_mz(formula, mz):
    left, right = formula.split('+')
    if '/' in left:
        left = int(left.split('/')[1])
        
    elif not left.startswith('M'):
        left = 1 / int(left[0])
    else:
        left = 1
    
    right = float(right)
    return (mz-right) * left

def multi_parse_mz(formulas, strucs, mz):
    return {struc: parse_mz(formula, mz) for struc, formula in zip(strucs, formulas)}
```

```python
def comb(a, length=2):
    N = 2**len(a)
    res = []
    for i in range(N):
        pivot = bin(i)[2:][::-1]
        re = []
        for j in range(len(pivot)):
            if int(pivot[j]):
                re.append(a[j])
        if len(re) == length:
        
            res.append(re)
        
    return res

#@deco
def find_parent_ion_by_adducts(df, ESI_mode='pos'):
    if ESI_mode == 'pos':
        table = pos_adds
    elif ESI_mode == 'neg':
        table = neg_adds
    else:
        print('ESI_mode param can just be either pos or neg!')
        return -1
    
    ms = {}
    if df.shape[0] == 1:
        return None
    
    mzs = df['mz'].values
    indices = df.index
    formulas = table[1]
    strucs = table[0]
    res = pd.DataFrame([multi_parse_mz(formulas, strucs, mz) for mz in mzs]).T
    res = res.rename({i:index for i, index in zip(range(len(indices)), indices)}, axis=1)
    
    c = comb(indices)
    for x1, x2 in c:
        #print(x1, x2)
        m = res[x1].values[:, None] - res[x2].values
        m = pd.DataFrame(m)
        m = m.apply(find_adducts, axis=1).dropna().values
        
        if len(m):
            ms['%s,%s' % (x1, x2)] = m

    if ms:
        return ms, res
    else:
        return None
```

```python
def find_adducts(x):
    l = len(x)
    res = np.array([-1, -1])
    
    for i, x0 in enumerate(x):
        pivot = np.array([x.name, i])
        if np.abs(x0) < 0.01:
            res = np.vstack([res, pivot])
          
    if res.size > 2:
        return res[1:, :]
```

```python
@deco
def multi_find_adducts(data):
    with Pool(processes=12) as pool:
        res = pool.map(find_parent_ion_by_adducts, data)
    return res
```

```python
res_after_find_adducts = multi_find_adducts(res_after_find_rt)
```

```python
from collections import defaultdict

def make_adducts_table(data):
    if data is None:
        return None
    
    dic, table = data
    raw_table = fd_final.loc[table.columns]
    res = defaultdict(list)
    for key, vals in dic.items():
        l, r = [int(i) for i in key.split(',')]
        for val in vals:
            val = np.ravel(val)
            l_strucs = table.index[val[0]]
            r_strucs = table.index[val[1]]
            
            l_mz = table.loc[l_strucs, l]
            r_mz = table.loc[r_strucs, r]
            res['l_index'].append(l)
            res['r_index'].append(r)
            res['l_originM/Z'].append(raw_table.loc[l, 'mz'])
            res['r_originM/Z'].append(raw_table.loc[r, 'mz'])
            res['l_struc'].append(l_strucs)
            res['r_struc'].append(r_strucs)
            res['l_M/Z'].append(l_mz)
            res['r_M/Z'].append(r_mz)
            res['M/Z_differ'].append(np.abs(l_mz-r_mz))
            res['rt_differ'].append(np.abs(raw_table.loc[l, 'rt'] - raw_table.loc[r, 'rt']))
    return pd.DataFrame(res)

@deco
def multi_make_adducts_table(data):
    with Pool(processes=6) as pool:
        res = pool.map(make_adducts_table, data)
        res = pd.concat(list(res))
        res = res.drop_duplicates()
        res.index = np.arange(res.shape[0])
    return res
```

```python
adducts_table = multi_make_adducts_table(res_after_find_adducts)
```

```python
adducts_table[['M/Z_differ', 'rt_differ']].corr()
```

```python
adducts_table.head()
```

```python
plt.figure(figsize=(12, 8))
sns.kdeplot(adducts_table['rt_differ'], adducts_table['M/Z_differ'], shade=True)
```

```python
plt.figure(figsize=(12, 8))
sns.kdeplot(adducts_table['rt_differ'], adducts_table['M/Z_differ'], shade=True, shade_lowest=False)
sns.scatterplot(adducts_table['rt_differ'], adducts_table['M/Z_differ'], color='steelblue', size=25, alpha=.5)
```

```python
sns.kdeplot(adducts_table['M/Z_differ'], shade=True)
```

```python
at = adducts_table[adducts_table['M/Z_differ'] < 0.005]
```

```python
plt.figure(figsize=(12, 8))
sns.kdeplot(at['rt_differ'], at['M/Z_differ'], shade=True)
```

```python
plt.figure(figsize=(12, 8))
sns.kdeplot(at['rt_differ'], at['M/Z_differ'], shade=True, shade_lowest=False)
sns.scatterplot(at['rt_differ'], at['M/Z_differ'], color='w', edgecolor='steelblue', size=30, legend=False)
```

总共多少对：

```python
(at['l_index'].astype('str').str[:] + at['r_index'].astype('str').str[:]).unique().size
```

```python
(adducts_table['l_index'].astype('str').str[:] + adducts_table['r_index'].astype('str').str[:]).unique().size
```

```python
adducts_table.shape
```

```python
at.to_csv('adducts.csv', index=None)
```

## Neutral Loss

```python
nloss = \
'''1 hydrogen H2 2.0157
2 water H2O 18.0106
3 methane CH4 16.0313
4 ethene C2H4 28.0313
5 ethine C2H2 26.0157
6 butene C4H8 56.0626
7 pentene C5H8 68.0626
8 benzene C6H6 78.0470
9 formaldehyde CH2O 30.0106
10 carbon monoxide CO 27.9949
11 formic acid CH2O2 46.0055
12 carbon dioxide CO2 43.9898
13 acetic acid C2H4O2 60.0211
14 ketene C2H2O 42.0106
15 propionic acid C3H6O2 74.0368
16 malonic acid C3H4O4 104.0110
17 malonic anhydride C3H2O3 86.0004
18 pentose equivalent C5H8O4 132.0423
19 deoxyhexose equivalent C6H10O4 146.0579
20 hexose equivalent C6H10O5 162.0528
21 hexuronic equivalent acid C6H8O6 176.0321 
22 ammonia NH3 17.0266
23 methylamine CH5N 31.0422
24 methylimine CH3N 29.0266
25 trimethylamine C3H9N 59.0735
26 cyanic acid CHNO 43.0058
27 urea CH4N2O 60.0324
28 phosphonic acid H3PO3 81.9820
29 phosphoric acid H3PO4 97.9769
30 metaphosphoric acid HPO3 79.9663
31 dihydrogen vinyl phosphate C2H5O4P 123.9926
32 hydrogen sulfide H2S 33.9877
33 sulfur S 31.9721
34 sulfur dioxide SO2 63.9619
35 sulfur trioxide SO3 79.9568
36 sulfuric acid H2SO4 97.9674 '''
```

```python
import re
```

```python
compound, structure, mass_weight = pd.DataFrame(nloss.split('\n'))[0].str.strip().str.findall('[\d]* (.*) (.*) (.*)').str[0].str
neutral_loss = pd.DataFrame([compound, structure, mass_weight]).T
neutral_loss.columns = ['compound', 'structure', 'mass weight']
neutral_loss['mass weight'] = neutral_loss['mass weight'].astype(float)
neutral_loss.head()
```

```python
def find_neutral_loss(s):
    for i, val in s.iteritems():
        val = np.abs(val)
        if val < 0.005:
            return True, i, val
    return None, None, None
```

```python
def find_parent_ion_by_neutral_loss(data):
    mzs = data['mz'].values
    indices = data.index
    d = {i:mz for i, mz in zip(indices, mzs)}
    res = []
    combs = comb(indices)
    for c in combs:
        # l is always smaller than r
        l, r = c
        temp = d[r] - (d[l] + neutral_loss['mass weight'])
        #return temp
        pivot, struc_i, sub = find_neutral_loss(temp)
        if pivot:
            df = pd.DataFrame([l, r, d[l], d[r], 
                                  neutral_loss['structure'][struc_i], 
                                  neutral_loss['mass weight'][struc_i],
                                  sub, np.abs(data.loc[l, 'rt']- data.loc[r, 'rt'])
                              ]).T
            df.columns = ['l_index', 'r_index', 'l_mz', 'r_mz', 
                          'nl_struc', 'mass weight', 'mz_differ', 'rt_differ']
            res.append(df)
    if res:
        return pd.concat(res)
    else:
        return None
```

```python
def multi_find_neutral_loss(data):
    with Pool(processes=12) as pool:
        res = pool.map(find_parent_ion_by_neutral_loss, data)
        return pd.concat(res).drop_duplicates()
```

```python
res_after_find_neutral_loss = multi_find_neutral_loss(res_after_find_rt)
```

```python
res_after_find_neutral_loss.head()
```

```python
plt.figure(figsize=(10, 6))
sns.kdeplot(res_after_find_neutral_loss['rt_differ'], res_after_find_neutral_loss['mz_differ'], shade=True, shade_lowest=False)
sns.scatterplot('rt_differ', 'mz_differ', data=res_after_find_neutral_loss)
```

```python
plt.figure(figsize=(10, 6))
sns.kdeplot(res_after_find_neutral_loss['rt_differ'], res_after_find_neutral_loss['mz_differ'], shade=True, shade_lowest=False)
sns.scatterplot('rt_differ', 'mz_differ', data=res_after_find_neutral_loss)
```

```python
plt.figure(figsize=(10, 6))
sns.kdeplot(res_after_find_neutral_loss['rt_differ'], res_after_find_neutral_loss['mz_differ'], shade=True, shade_lowest=False, cmap='Blues')
sns.scatterplot('rt_differ', 'mz_differ', data=res_after_find_neutral_loss)
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
plt.hlines(0.001, -.5, 1.5, colors='navy', linestyles='dashed', alpha=.8)
plt.vlines(0.5, -1, 1, colors='goldenrod', linestyles='dashed', alpha=.8)
plt.xlim(xlim)
plt.ylim(ylim)
plt.text(-.1, 0.0012, '↓0.001↓', ha='center', va='center', size=20, color='navy')
plt.text(.6, 0.0012, '←0.5←', ha='center', va='center', size=20, color='goldenrod')
```

```python
nl = res_after_find_neutral_loss[(res_after_find_neutral_loss['mz_differ'] < 0.001) & (res_after_find_neutral_loss['rt_differ'] < .5)]
```

```python
nl['nl_struc'].value_counts()
```

```python
nl.to_csv('neutral_loss.csv', index=None)
```

## Aggregation

```python
res_after_find_neutral_loss['nl_struc'].value_counts()
```

```python
adducts_table['l_struc'].value_counts()[:5]
```

```python
adducts_table['r_struc'].value_counts()[:5]
```

```python
at['l_struc'].value_counts()[:5]
```

```python
at['r_struc'].value_counts()[:5]
```

```python
at['r_struc'].value_counts()[:5]
```

```python
a = dict(at['l_struc'].value_counts())
b = dict(at['r_struc'].value_counts())
```

```python
for c in list(b):
    if c in a:
        a[c] += b[c]
    else:
        a[c] = b[c]
```

```python
pd.Series(a)
```

```python
a = adducts_table[(adducts_table['l_struc'].str.contains('ACN')) | (adducts_table['r_struc'].str.contains('ACN'))]
```

```python
np.intersect1d((a['l_index'].values),(a['r_index'].values))
```

```python
adducts_table[adducts_table['l_index'] == 872]
```

## all indices that involved in iso, adducts or neutral loss

```python
iso_is = set(iso_table['fd_index'])
len(iso_is)
```

```python
adducts_is = set(adducts_table['l_index'].values)
adducts_is.update(adducts_table['r_index'].values)
len(adducts_is)
```

```python
nl_is = set(res_after_find_neutral_loss['l_index'].values)
nl_is.update(res_after_find_neutral_loss['r_index'])
len(nl_is)
```

```python
from matplotlib_venn import venn3
```

```python
plt.figure(figsize=(10, 6))
venn3([iso_is, adducts_is, nl_is], ('Isotope', 'Adduct', 'neutral loss'))
```

```python
all_is = set.union(iso_is, adducts_is, nl_is)
single_fd = fd_final.copy().loc[(fd_final.index ^ pd.Index(all_is))]
into_agg = into.copy().loc[fd_final.index]
fd_need_agg = fd_final.copy().loc[pd.Index(all_is)]
fd_need_agg['isoadnl'] = 0
fd_need_agg['drop'] = 1
into_agg.shape, fd_need_agg.shape
```

```python
def agg_iso(iso_table, fd_table, into_table, how='sum'):
    fd_need_agg = fd_table.copy()
    into_agg = into_table.copy()
    cols = into_agg.columns
    if not (how == 'sum' or how == 'drop'):
        raise TypeError('agg_iso() got an unexpected keyword argument "how" > ("sum" or "drop")')
    prime_list = []
    for group in iso_table['group'].unique():
        temp = iso_table[iso_table['group']==group].copy()
        prime_list.append(temp.iloc[-1, 4])
    
    for group in iso_table['group'].unique():
        temp = iso_table[iso_table['group']==group].copy()
        if temp.iloc[0, 0] < temp.iloc[-1,0]:
            raise IndexError('invalid order in group:%s' % group)
        prime = temp.index[-1]
        free = temp.index[:-1]
        
        total = temp.index
        
        pivot = False
        for f in free:
            if not f in prime_list:
                fd_need_agg.loc[f, 'drop'] = np.nan
                pivot = True
        if pivot:
            fd_need_agg.loc[prime, 'isoadnl'] = 1
        
        if how=='sum':
            for f in free:
                if not f in prime_list:
                    into_agg.loc[prime, cols[1:]] = into_agg.loc[prime, cols[1:]] + into_agg.loc[f, cols[1:]]
    fd_need_agg = fd_need_agg.dropna()
    into_agg = into_agg.loc[fd_need_agg.index]
    return fd_need_agg, into_agg
```

```python
def agg_iso(iso_table, fd_table, into_table, how='sum', conservative=True):
    fd_need_agg = fd_table.copy()
    into_agg = into_table.copy()
    cols = into_agg.columns
    if not (how == 'sum' or how == 'drop'):
        raise TypeError('agg_iso() got an unexpected keyword argument "how" > ("sum" or "drop")')
    prime_list = []
    for group in iso_table['group'].unique():
        temp = iso_table[iso_table['group']==group].copy()
        prime_list.append(temp.iloc[-1, 4])
    
    for group in iso_table['group'].unique():
        temp = iso_table[iso_table['group']==group].copy()
        if temp.iloc[0, 0] < temp.iloc[-1,0]:
            raise IndexError('invalid order in group:%s' % group)
        prime = temp.index[-1]
        free = temp.index[:-1]
        
        total = temp.index
        
        pivot = False
        
        val = fd_need_agg.loc[prime, 'var']
        for f in free:
            if conservative and fd_need_agg.loc[f, 'var'] > val:
                break
            if not f in prime_list:
                fd_need_agg.loc[f, 'drop'] = np.nan
                pivot = True
            if how=='sum':
                if not f in prime_list:
                    into_agg.loc[prime, cols[1:]] = into_agg.loc[prime, cols[1:]] + into_agg.loc[f, cols[1:]]
        if pivot:
            fd_need_agg.loc[prime, 'isoadnl'] = 1
        
    fd_need_agg = fd_need_agg.dropna()
    into_agg = into_agg.loc[fd_need_agg.index]
    return fd_need_agg, into_agg
```

```python
fd_iso, into_iso = agg_iso(iso_table, fd_need_agg, into_agg, conservative=True)
```

```python
fd_iso['isoadnl'].value_counts()
```

```python
fd_iso.shape
```

```python
def agg_ad(fd_iso, into_iso, adducts_table):
    fd_ad = fd_iso.copy()
    into_ad = into_iso.copy()
    cols = into_ad.columns
    indices = fd_ad.index
    pairs = at[['l_index', 'r_index']].drop_duplicates().values
    fd_ad['pivot'] = 0
    for pair in pairs:
        if pair[0] in indices and pair[1] in indices:
            temp = fd_ad.loc[pair]
            fd_ad.loc[temp.index[-1], 'drop'] = np.nan
            into_ad.loc[temp.index[0], cols[1:]] += into_ad.loc[temp.index[-1], cols[1:]]
            fd_ad.loc[temp.index[0], 'pivot'] = 2
            
    fd_ad['isoadnl'] = fd_ad['isoadnl'] + fd_ad['pivot']
    fd_ad = fd_ad.drop('pivot', axis=1)
    fd_ad = fd_ad.dropna()
    into_ad = into_ad.loc[fd_ad.index]

    return fd_ad, into_ad
```

```python
fd_ad, into_ad = agg_ad(fd_iso, into_iso, adducts_table)
```

```python
fd_ad.shape, into_ad.shape
```

```python
fd_ad['isoadnl'].value_counts()
```

```python
def agg_nl(fd_ad, into_ad, res_after_find_neutral_loss):
    fd_nl = fd_ad.copy()
    into_nl = into_ad.copy()
    cols = into_nl.columns
    indices = fd_nl.index
    fd_nl['pivot'] = 0
    pairs = res_after_find_neutral_loss[['l_index', 'r_index']].values
    for pair in pairs:
        if pair[0] in fd_nl.index and pair[1] in indices:
            temp = fd_nl.loc[pair]
            fd_nl.loc[temp.index[-1], 'drop'] = np.nan
            into_nl.loc[temp.index[0], cols[1:]] += into_nl.loc[temp.index[-1], cols[1:]]
            fd_nl.loc[temp.index[0], 'pivot'] = 4
    fd_nl['isoadnl'] = fd_nl['isoadnl'] + fd_nl['pivot']
    fd_nl = fd_nl.drop('pivot', axis=1)
    fd_nl = fd_nl.dropna()
    into_nl = into_nl.loc[fd_nl.index]
    return fd_nl, into_nl
```

```python
fd_nl, into_nl = agg_nl(fd_ad, into_ad, res_after_find_neutral_loss)
```

```python
fd_nl.shape, into_nl.shape
```

```python
fd_nl['isoadnl'].value_counts()
```

```python
fd_nl = fd_nl.drop('drop', axis=1)
```

```python
single_fd['isoadnl'] = 0
```

```python
fd_after_agg = pd.concat([single_fd, fd_nl])
```

```python
as_after_agg = allsamples.loc[fd_after_agg.index]
```

```python
fagg = as_after_agg.T.values

fagg = fagg / (fagg.sum(1) / fagg.sum(1).max())[:, np.newaxis]
fagg = StandardScaler().fit_transform(fagg)
```

```python
pca_mat = draw_pca(fagg)
```

```python
draw_trend(pca_mat)
```

# Variable selection

```python
VIPs = pd.read_csv('vip_from_pls_da.csv', index_col=0)

vip_index = VIPs[(VIPs['comp 1'] > 1)&(VIPs['comp 2'] > 1)].index
vip_index = set(vip_index)
```

```python
volcano = pd.DataFrame([pvalues_fdr_bh, np.abs(times)]).T
volcano.index = a.columns
volcano.columns = ['p_val', 'fold_change']
t_index = set(volcano[(volcano['p_val'] < .05)&(volcano['fold_change'] > 2)].index)
```

```python
rf_index = set(fi[fi['c'] == 1].index.astype(float))
```

```python
from matplotlib_venn import venn3_circles
```

```python
plt.figure(figsize=(10,10))
cs = ('lightskyblue', 'darkred', 'forestgreen')
v = venn3((vip_index, t_index, rf_index), ('VIP', 'Volcano\nplot', 'RandomForest'), cs)
c = venn3_circles((vip_index, t_index, rf_index), linestyle='solid', linewidth=2, color='grey')
for p in v.subset_labels:
    if p:
        plt.setp(p, fontsize=18)
for i, p in enumerate(v.set_labels):
    if p:
        plt.setp(p, fontsize=18, color=cs[i], fontweight='bold', fontstyle='italic', ha='center')
plt.gca().set_facecolor('seashell')
plt.gca().set_axis_on()
plt.savefig('venn3.png')
```

```python
pca_mat = draw_pca(f)
plt.close()
```

```python
total_ints = allsamples.T
total_ints.index = names
```

```python
total_ints = total_ints.loc[~total_ints.index.str.startswith('QC-eq')]
```

```python
total_ints = pd.DataFrame(total_ints.sum(1))
```

```python
import scipy.stats as stats
```

```python
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h
```

```python
total_data = total_ints.pivot_table(0, index=total_ints.index, aggfunc=('mean', 'std', 'min', 'max', mean_confidence_interval))['mean_confidence_interval']
total_data = pd.DataFrame(total_data)
```

```python
total_data['mean'], total_data['down'], total_data['up'] = total_data['mean_confidence_interval'].str[0], total_data['mean_confidence_interval'].str[1], total_data['mean_confidence_interval'].str[2]
```

```python
total_data = total_data.drop('mean_confidence_interval', axis=1)
total_data
```

```python
total_ints['cate'] = total_ints.index
total_ints = total_ints.rename({0:'int'}, axis=1)
```

```python
groups = ['a', 'b', 'c', 'd', 'e', 'c']
```

```python
g = sns.catplot(x='cate', y='int', data=total_ints, kind='bar',
           palette='muted')
fig = g.fig
fig.set_size_inches(12, 6)
g.despine(left=True)
axes = fig.axes[0]
axes.set_xticklabels(['Sample4', 'Sample1', 'QC', 'Sample5', 'Sample3', 'Sample2'], fontsize=15)
axes.set_xlabel('Class', fontsize=18)
axes.set_ylabel('Total Intensities', fontsize=18)
for i, g in enumerate(groups):
    plt.text(i, total_data.loc[['Sa-4', 'Sa-1', 'QC-nor', 'Sa-5', 'Sa-3', 'Sa-2']].iloc[i,0]+5e9, g, 
            ha='center', va='center', fontsize=15, fontweight='bold')
plt.savefig('total_intensities.png')
```

```python
fd_after_agg.shape
```

```python
fig = plt.figure(figsize=(12,6))
sns.scatterplot(x='rt', y='mz', data=fd_after_agg[fd_after_agg['c']==0], alpha=.2, )
sns.scatterplot(x='rt', y='mz', data=fd_after_agg[fd_after_agg['c']==1], size='var')
plt.xlim(0, 1000)
plt.ylim(50, 1000)
#plt.savefig('scatter_rf_agg.png')
```

```python
fig = plt.figure(figsize=(12,6))
sns.scatterplot(x='rt', y='mz', data=fd_after_agg[fd_after_agg['c']==0], alpha=.2, )
sns.scatterplot(x='rt', y='mz', data=fd_after_agg[fd_after_agg['c']==1], size='var')
plt.xlim(0, 1000)
plt.ylim(50, 1000)
plt.savefig('scatter_rf_agg.png')
```

```python
fig = plt.figure(figsize=(12,6))
sns.scatterplot(x='rt', y='mz', data=fd_need_agg[fd_need_agg['c']==0], alpha=.2, )
sns.scatterplot(x='rt', y='mz', data=fd_need_agg[fd_need_agg['c']==1], size='var')
plt.xlim(0, 1000)
plt.ylim(50, 1000)
#plt.savefig('scatter_rf_agg.png')
```

```python
fig = plt.figure(figsize=(12,6))
sns.scatterplot(x='rt', y='mz', data=single_fd[single_fd['c']==0], alpha=.2, )
sns.scatterplot(x='rt', y='mz', data=single_fd[single_fd['c']==1], size='var')
plt.xlim(0, 1000)
plt.ylim(50, 1000)
#plt.savefig('scatter_rf_agg.png')
```

```python
fig = plt.figure(figsize=(12,6))
sns.scatterplot(x='rt', y='mz', data=fd_final[fd_final['c']==0], alpha=.2, )

sns.scatterplot(x='rt', y='mz', data=single_fd[single_fd['c']==1], size='var', legend=False)
sns.scatterplot(x='rt', y='mz', data=fd_need_agg[fd_need_agg['c']==1], size='var', legend=False)
temp_index = (fd_need_agg.index ^ fd_nl.index)
temp_fd = fd_final.loc[temp_index]
sns.scatterplot(x='rt', y='mz', data=temp_fd[temp_fd['c']==1], size='var', legend=False)
ax=plt.gca()
l = ['Not important variables', 'Monoions', 'Fused ions', 'Ions dropped after fusion']
for i, p in enumerate(ax.get_children()[:4]):
    color = p.get_facecolors()
    plt.text(750, 900-i*50, l[i], fontsize=15,
            ha='left', va='center')
    plt.scatter(700, 900-i*50, c=color, s=100)
    
plt.xlim(0, 1000)
plt.ylim(50, 1000)
plt.savefig('scatter_rf_all.png')
```

```python
fd_need_agg.shape, fd_nl.shape
```

```python
def find(rt=None, mz=None, thld=1):
    fd = fd_final.iloc[:, [1,4,-4,-2,-1]].copy()
    if mz:
        fd = fd[(fd['mz'] > mz-thld/2)&(fd['mz'] < mz+thld/2)]
    if rt:
        return fd[(fd['rt'] > rt-thld/2) & (fd['rt'] < rt+thld/2)]
    if fd == fd_final:
        return None
```

```python
QCs_nor.loc[[2187, 2407]].mean(1)
```

```python
i = find(644, thld=1)
i
```

```python
iso_table.loc[i.index & iso_table.index]
```

```python
adducts_table[np.isin(adducts_table['l_index'], i.index) | np.isin(adducts_table['r_index'], i.index)]
```

```python
res_after_find_neutral_loss[np.isin(res_after_find_neutral_loss['l_index'], i.index) | np.isin(res_after_find_neutral_loss['r_index'], i.index)]
```

```python
i
```

```python
plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=(12,6))
sns.scatterplot(x='rt', y='mz', data=i, hue='c', size='var')
plt.xlim(643.5, 645)
plt.ylim(150, 400)
#plt.vlines(i['rt'].mean(), 0,1000, linestyles='dashed', colors='coral', alpha=.5)
#plt.vlines(i['rt'].mean()-.5, 0,1000, linestyles='solid', colors='lightcoral')
#plt.vlines(i['rt'].mean()+.5, 0,1000, linestyles='solid', colors='lightcoral')
for vals in i.iterrows():
    plt.text(vals[1]['rt']+.02, vals[1]['mz'], vals[0], ha='left', va='center')
#plt.grid(which='both', alpha=.5)

```

```python
pd.Series(np.concatenate([adducts_table['l_struc'].values, adducts_table['r_struc'].values])).value_counts()
```

```python
res_after_find_neutral_loss['nl_struc'].value_counts()
```

# More and more..

```python
qc = QCs_nor.loc[pass_index].copy()
qc['RSD'] = qc.apply(lambda x: np.std(x, ddof=1)/np.mean(x), axis=1)
qc['median'] = qc.median(1)
qc['std'] = np.std(qc.iloc[:, :7], 1, ddof=1)
qc = qc.dropna(subset=['std'])
qc['rank'] = qc['median'].argsort().argsort()

fig, ax = plt.subplots(figsize=(12,6))
sns.scatterplot('rank', 'std', data=qc)
LOESS = sm.nonparametric.lowess(qc['std'], qc['rank'])
ax.plot(LOESS[:, 0], LOESS[:, 1], color='r', linewidth=2, linestyle='solid')
```

```python
qc = QCs_nor.loc[pass_index].copy()
qc['RSD'] = qc.apply(lambda x: np.std(x, ddof=1)/np.mean(x), axis=1)
qc['median'] = qc.median(1)
qc['std'] = np.std(qc.iloc[:, :7], 1, ddof=1)
qc = qc.dropna(subset=['std'])
qc['rank'] = qc['median'].argsort().argsort()

fig, [ax, ax2] = plt.subplots(1, 2, figsize=(12,6))
sns.scatterplot('median', 'std', data=qc, ax=ax)
LOESS = sm.nonparametric.lowess(qc['std'], qc['median'])
ax.plot(LOESS[:, 0], LOESS[:, 1], color='k', linewidth=2, linestyle='dashed')
ax2.plot(LOESS[:, 0], LOESS[:, 1])
```

```python
qc = QCs_nor.loc[pass_index].copy()
qc['RSD'] = qc.apply(lambda x: np.std(x, ddof=1)/np.mean(x), axis=1)
qc['median'] = np.log(qc.median(1))
qc['std'] = np.std(qc.iloc[:, :7], 1, ddof=1)
qc = qc.dropna(subset=['std'])
qc['rank'] = qc['median'].argsort().argsort()

fig, [ax,ax2] = plt.subplots(1,2,figsize=(12,6))
sns.scatterplot('median', 'std', data=qc, ax=ax)
LOESS = sm.nonparametric.lowess(qc['std'], qc['median'])
ax.plot(LOESS[:, 0], LOESS[:, 1], color='k', linewidth=2, linestyle='dashed')
ax2.plot(LOESS[:,0], LOESS[:, 1])
```

```python
plt.plot(LOESS[:,0], LOESS[:,1])
```

```python
qc.head()
```

```python
qc = QCs_nor.loc[pass_index].copy()
qc['RSD'] = qc.apply(lambda x: np.std(x, ddof=1)/np.mean(x), axis=1)
qc['median'] = qc.median(1)
qc['std'] = np.std(qc.iloc[:, :7], 1, ddof=1)
qc = qc.dropna(subset=['std'])
qc['rank'] = qc['median'].argsort().argsort()

fig, ax = plt.subplots(figsize=(12,6))
sns.scatterplot('rank', 'RSD', data=qc)
#LOESS = sm.nonparametric.lowess(qc['std'], qc['rank'])
#ax.plot(LOESS[:, 0], LOESS[:, 1], color='k', linewidth=2, linestyle='dashed')
```

```python
qc = np.log(QCs_nor.loc[pass_index].copy())
qc['RSD'] = qc.apply(lambda x: np.std(x, ddof=1)/np.mean(x), axis=1)
qc['median'] = qc.median(1)
qc['std'] = np.std(qc.iloc[:, :7], 1)
qc = qc.dropna(subset=['std'])
qc['rank'] = qc['median'].argsort().argsort()

fig, ax = plt.subplots(figsize=(12,6))
sns.scatterplot('rank', 'std', data=qc)
LOESS = sm.nonparametric.lowess(qc['std'], qc['rank'])
ax.plot(LOESS[:, 0], LOESS[:, 1], color='k', linewidth=2, linestyle='dashed')
```

## Pseudo imitation

```python
qc = QCs_nor.loc[pass_index].copy()
qc['RSD'] = qc.apply(lambda x: np.std(x, ddof=1)/np.mean(x), axis=1)
qc['median'] = qc.median(1)
qc['std'] = np.std(qc.iloc[:, :7], 1, ddof=1)
qc = qc.dropna(subset=['std'])
qc['rank'] = qc['median'].argsort().argsort()

fig, ax = plt.subplots(figsize=(12,6))
sns.scatterplot('rank', 'std', data=qc)
LOESS = sm.nonparametric.lowess(qc['RSD'], qc['rank'])
ax.plot(LOESS[:, 0], LOESS[:, 1], color='k', linewidth=2, linestyle='dashed')
```

```python
qc = QCs_nor.loc[pass_index].copy()
tqc = qc['QC_3_3.mzXML'].copy()
for i in range(7):
    qc.iloc[:, i] = np.log2(qc.iloc[:, i] / tqc)

fig, ax = plt.subplots(figsize=(8,6))
ax.set_ylim(-2,2)
sns.boxplot('variable', 'value', data=qc.melt(value_vars=qc.columns[:7]), linewidth=1, ax=ax)
ax.tick_params('x', labelrotation=30)
```

```python
qc['cv'] = pd.cut(qc['RSD'], bins=(0, 0.1, 0.15, 0.20, 0.25, 0.30, 100), labels=('<10%', '10~15%', '15~20%', '20~25%', '25~30%', '>30'))
```

```python
sns.countplot('cv', data=qc, palette='winter_r')
```

```python
ft = filters.iloc[:, 1:]
ft = ft.loc[allsamples.index]
```

```python
combs = comb(ft.columns)
fig, axs = plt.subplots(8, 2, figsize=(18,40))
axs = axs.ravel()
for i, c in enumerate(combs):
    ax = axs[i]
    ax.set_facecolor('floralwhite')
    ftt = pd.DataFrame((ft[c[0]]) / (ft[c[1]])).rename({0:'fd'}, axis=1)
    ftt['fd'] = np.log2(ftt['fd'])
    ftt['rank'] = QCs_nor.mean(1).fillna(0).argsort().argsort()
    ftt['index'] = ftt.index
    ftt['median'] = np.log(QCs_nor.median(1))
    sns.scatterplot('median', 'fd', data=ftt, ax=ax)
    
    LOESS = sm.nonparametric.lowess(ftt['fd'], ftt['median'])
    ax.plot(LOESS[:, 0], LOESS[:, 1], color='k', linewidth=2, linestyle='dashed')
    ax.set_xticks([])
    ax.set_xlabel('')
    change = np.log2(eval(c[0]+'/'+c[1]))
    ax.set_ylim(change-2, change+2)
    ax.text(20, change+1, 'Theory: %s\n%s<>%s' % (round(change,2), c[0], c[1]), fontsize=18, color='r')
    ax.hlines(change, 0, 50, color='r')
    ax.set_xlim(ftt['median'].min()-1, ftt['median'].max()+1)
    print(c, 'DONE!', end='----')
plt.tight_layout()
```

```python
a = allsamples.join(filters.iloc[:, 1:]).fillna(0)
```

```python
a = into.iloc[:, 1:]

a = a.iloc[:, [11, 6, 10, 16, 8, 19, 18, 9, 
          31, 1, 22, 12, 
          34, 28, 23, 7, 
          33, 25, 35, 20, 
          3, 29, 30, 14, 
          4, 26, 32, 13, 
          0, 2, 21, 17, 
          24, 5, 27, 15]]
```

```python
def deal_names2(cols):
    re = []
    for i, name in enumerate(cols):
        if i in [0, 1, 2]:
            re.append('QC-eq-%s' % str(i+1))
        elif i in [3, 4, 5, 6, 7]:
            re.append('QC-eq')
            
        elif name.startswith('Sa'):
            name = name[:10]
            re.append('Sa-%s' % name[-3])
            
        elif name.startswith('QC'):
            re.append('QC-nor')
        else:
            import re as r
            name = r.findall('.*_(\d*)', name)[0]
            name = list('{:0<3}'.format(name))
            name.insert(1, '.')
            name = ''.join(name)
            
            re.append('Fi%s' % name)
    
    return re

def NOR_total_ion(mat):
    p = np.median(mat[[11, 15, 19, 23, 27, 31, 35], :].sum(1))
    return mat / ((mat.sum(1) / p)[:, np.newaxis])

def NOR_median(mat):
    p = np.median(mat[[11, 15, 19, 23, 27, 31, 35], :], 0)
    p = np.median(mat / p, 1)
    return mat / p[:, np.newaxis]

def NOR_loess(mat):
    pass
def draw_pca2(df, normalization_method='median', order=True):
    df = df.dropna(how='all').fillna(10)
    mat = df.T.values
    
    if normalization_method=='median':
        mat = NOR_median(mat)
    elif normalization_method=='total_ion':
        mat = NOR_total_ion(mat)
    elif normalization_method==None:
        pass
    else:
        print('invalid method string, skip normalization')
    #mat = ((mat - mat.mean(0)) / ((mat.std(0, ddof=1))**1))
    PCA_allsamples = PCA(n_components=5, whiten=False).fit_transform(mat)
    names = deal_names2(df.columns)
    colormap = {'QC-eq-1': 'brown', 
                'QC-eq-2': 'darkred', 
                'QC-eq-3': 'red', 
                'QC-eq': 'indianred', 
                'QC-nor': 'lightcoral', 
                'Sa-1': 'plum', 
                'Sa-2': 'skyblue', 
                'Sa-3': 'teal', 
                'Sa-4': 'forestgreen', 
                'Sa-5': 'peru', 
                'Fi0.10': '#E8E8E8',
                'Fi0.15': '#CFCFCF',
                'Fi0.30': '#B5B5B5',
                'Fi0.50': '#9C9C9C',
                'Fi0.75': '#4F4F4F',
                'Fi1.00': '#1C1C1C'
               
               }
    pca_mat = pd.DataFrame(PCA_allsamples)

    pca_mat['class'] = names
    pca_mat['cate'] = pca_mat['class'].str[:2]
    
    fig, axs = plt.subplots(1,2, figsize=(16,6))
    
    ax = axs[0]
    scatters = sns.scatterplot(0, 1, hue='class', style='cate', data=pca_mat, s=200, ax=ax, palette=colormap, alpha=.7)
    if order:
        for i in range(8, 36):
            if names[i] == 'QC-nor':
                continue
            ax.text(pca_mat.iloc[i, 0]+2, pca_mat.iloc[i, 1], i, fontsize=15, ha='left', va='center')
    ax.legend(fontsize=10, bbox_to_anchor=[1,1, .2, 0])
    ax.set_xlabel('Component 1', fontsize=18)
    ax.set_ylabel('Component 2', fontsize=18) 
    
    ax = axs[1]
    sns.scatterplot(0, 1, hue='class', style='cate', data=pca_mat[(pca_mat['cate']=='QC') | (pca_mat['cate'] == 'Fi')], s=200, ax=ax, palette=colormap)
    ax.legend(fontsize=10)
    ax.set_xlabel('Component 1', fontsize=18)
    ax.set_ylabel('Component 2', fontsize=18)
    plt.tight_layout()
    return pca_mat, mat
```

```python
s, d = draw_pca2(a.loc[pass_index], normalization_method='median')
```

```python
QCs_nor = QCs_nor.loc[pass_index]
```

```python
m = QCs_nor.median(1)

fa = (QCs_nor / m[:, None]).median().values

qcs = QCs_nor / fa
```

```python
qcs = QCs_nor / (QCs_nor.mean() / QCs_nor.mean().max())
```

```python
orsd = QCs_nor.apply(lambda x: np.std(x, ddof=1)/np.mean(x), 1)
```

```python
nrsd = qcs.apply(lambda x: np.std(x, ddof=1) / np.mean(x), 1)
```

```python
sns.kdeplot(orsd, shade=True, label='old')
sns.kdeplot(nrsd, shade=True, label='new')
plt.legend()
```

```python
a = a.loc[pass_index]
```

```python
a0 = np.log2((a/a.median(1)[:, None]))
a1 = a / (a/a.median(1)[:,None]).median()
a1 = np.log2(a1/a1.median(1)[:, None])
```

```python
plt.figure(figsize=(25,5))
sns.boxplot(x='variable', y='value', data=a0.melt(), linewidth=.5)
plt.ylim(-4, 4)
plt.gca().tick_params('x', rotation=30)
```

```python
plt.figure(figsize=(25,5))
sns.boxplot(x='variable', y='value', data=a1.melt(), linewidth=.5)
plt.ylim(-4, 4)
plt.gca().tick_params('x', rotation=30)
```

```python
a2 = pd.DataFrame(d).T
a2.index = a1.index
a2.columns = a1.columns
a2 = np.log2(a2/a2.median(1)[:, None])
```

```python
plt.figure(figsize=(25,5))
sns.boxplot(x='variable', y='value', data=a2.melt(), linewidth=.5)
plt.ylim(-4, 4)
plt.gca().tick_params('x', rotation=30)
```

```python
loess = sm.nonparametric.lowess
```
