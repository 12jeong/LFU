%reset -f
#%% import
import os
os.getcwd()
from os import chdir
#os.chdir('C:\\Users\\UOS')
os.chdir('C:\\Users\\MYCOM')
pdir = os.getcwd() ;print(pdir)

# import sys
import pandas as pd
import seaborn as sns # 시각화
import matplotlib.pyplot as plt
# 그래프의 스타일을 지정
plt.style.use('ggplot')
import numpy as np
import scipy as sp
import sklearn
import matplotlib as mpl
mpl.rcParams.update({'font.size':14})
plt.rc('font',family='Malgun Gothic') # windows
%matplotlib inline 
import matplotlib.pyplot as plt

#%% expand output display pd.df
pd.set_option('display.expand_frame_repr', False) 

#%% Load raw data
df_buy  = pd.read_csv(".\Dropbox\\LFY\\datasets\\ppdata\\df_buy.csv") 
df_nobuy = pd.read_csv(".\Dropbox\\LFY\\datasets\\ppdata\\df_nobuy.csv")
online_bh = pd.read_csv(".\Dropbox\\LFY\\datasets\\ppdata\\online_bh.csv")
trans_info = pd.read_csv(".\Dropbox\\LFY\\datasets\\ppdata\\trans_info.csv")
df_design_buy = pd.read_csv(".\Dropbox\\LFY\\datasets\\ppdata\\df_design_buy.csv",index_col=0)
mg_ppdata = pd.read_csv(".\Dropbox\\LFY\\datasets\\ppdata\\mg_ppdata.csv")

               
#%% df_buy 의 연속형변수 군집화
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# continuos variable
X = df_design_buy[df_design_buy.buy == 1]
X = X.drop(['sess_dt','sess_id','clnt_id','id','biz_unit','clnt_age','clnt_gender','dvc_ctg_nm','trfc_src','buy'],axis=1)
X = X.dropna()
X_col = X.columns

# In general, it's a good idea to scale the data prior to PCA.
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)    
pca = PCA()
x_new = pca.fit_transform(X)
pca.components_[0] # 1주성분
pca.components_[1] # 2주성분

pd.DataFrame([X_col,pca.components_[0],pca.components_[1]]).transpose()
plt.plot(np.cumsum(pca.explained_variance_ratio_))

# kmeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:,0],X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5);

# plot PCA loading and loading in biplot
y = y_kmeans
def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

myplot(x_new[:,0:2],np.transpose(pca.components_[0:2, :]))
plt.show()

#%% A02
df_design_buy.head()
df_A02 = df_design_buy[(df_design_buy.biz_unit=="A02")&(df_design_buy.buy==1)].drop_duplicates('clnt_id').drop(['biz_unit','buy'],axis=1)
df_A02_tmp = pd.melt(df_A02, id_vars=['clnt_id','id','clnt_age','dvc_ctg_nm','clnt_gender','sess_dt','sess_id','tot_pag_view_ct','tot_seq','tot_sess_hr_v','trfc_src'])

sns.catplot(x="clnt_gender", y="action_count0", hue="clnt_age", col='variable', data=df_A02, kind="bar",height=4, aspect=.7)
