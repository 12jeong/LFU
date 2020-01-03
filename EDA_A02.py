%reset -f
#%% import
import os
os.getcwd()
from os import chdir
# os.chdir('C:\\Users\\UOS')
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

#plt.scatter(X[:,0],X[:, 1], c=y_kmeans, s=50, cmap='viridis')
#centers = kmeans.cluster_centers_
#plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5);

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
plt.show(

#%% A02
df_design_buy.head()
df_A02 = df_design_buy[(df_design_buy.biz_unit=="A02")&(df_design_buy.buy==1)].drop_duplicates('clnt_id').drop(['biz_unit','buy'],axis=1)

# 연령대별 action_count와 action_time 분포의 차이
df_A02_tmp = pd.melt(df_A02, id_vars=['clnt_id','id','clnt_age','dvc_ctg_nm','clnt_gender','sess_dt','sess_id','tot_pag_view_ct','tot_seq','tot_sess_hr_v','trfc_src'])

df_A02_count = df_A02_tmp[df_A02_tmp['variable'].str.contains( "count" )] 
sns.catplot(x="variable", y='value', hue="clnt_age" , col="clnt_age", data=df_A02_count , kind="bar",height=4, aspect=.7)
sns.catplot(x="variable", y='value', hue="trfc_src" , col="trfc_src", data=df_A02_count , kind="bar",height=4, aspect=.7)

df_A02_time = df_A02_tmp[df_A02_tmp['variable'].str.contains( "time" )] 
sns.catplot(x="variable", y='value', hue="clnt_age", col="clnt_age", data=df_A02_time , kind="bar",height=4, aspect=.7)
sns.catplot(x="variable", y='value', hue="trfc_src" , col="trfc_src", data=df_A02_time , kind="bar",height=4, aspect=.7)

#%%
len(pd.unique(df_A02.clnt_id))

df_A02.columns
df_A02_by_clnt = df_A02.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18]].groupby(['clnt_id','clnt_age']).agg('mean').reset_index() 
df_A02_tmp = pd.melt(df_A02_by_clnt, id_vars=['clnt_id','clnt_age'])

df_A02_by_clnt_time = df_A02_tmp[df_A02_tmp['variable'].str.contains("time")] 
sns.catplot(x="variable", y='value', hue="clnt_age", col="clnt_age", data=df_A02_by_clnt_time , kind="bar",height=4, aspect=.7)

df_A02_by_clnt_count = df_A02_tmp[df_A02_tmp['variable'].str.contains("count")] 
sns.catplot(x="variable", y='value', hue="clnt_age", col="clnt_age", data=df_A02_by_clnt_count , kind="bar",height=4, aspect=.7)

#%% 각연령, 나이대, 평일/주말에 뭘 많이 사는지
trans_A02 = trans_info[trans_info.biz_unit == "A02"]

s = trans_A02.groupby(['clac_nm1']).clnt_age.value_counts(normalize=True).mul(100)
plt.rcParams["figure.figsize"] = (20,15)
s.unstack().plot.barh(stacked=True)

plt.rcParams["figure.figsize"] = (10,10)
trans_A02[trans_A02.clnt_age==10].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.clnt_gender=="F")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.clnt_gender=="M")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.weekend==0)].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.weekend==1)].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="F")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="M")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="F")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="M")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)


trans_A02[trans_A02.clnt_age==20].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.clnt_gender=="F")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.clnt_gender=="M")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.weekend==0)].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.weekend==1)].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="F")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="M")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="F")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="M")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)


trans_A02[trans_A02.clnt_age==30].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.clnt_gender=="F")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.clnt_gender=="M")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.weekend==0)].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.weekend==1)].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="F")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="M")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="F")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="M")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)

trans_A02[trans_A02.clnt_age==40].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.clnt_gender=="F")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.clnt_gender=="M")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.weekend==0)].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.weekend==1)].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="F")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="M")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="F")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="M")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)


trans_A02[trans_A02.clnt_age==50].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.clnt_gender=="F")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.clnt_gender=="M")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.weekend==0)].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.weekend==1)].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="F")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="M")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="F")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="M")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)


trans_A02[trans_A02.clnt_age==60].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.clnt_gender=="F")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.clnt_gender=="M")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.weekend==0)].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.weekend==1)].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="F")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="M")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="F")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="M")].clac_nm1.value_counts(normalize=True).mul(100).plot.barh(stacked=True)



#%% 구체적인 소분류는 뭘 사는지 -2
trans_A02 = trans_info[trans_info.biz_unit == "A02"]
len(pd.unique(trans_A02.clac_nm1))
len(pd.unique(trans_A02.clac_nm2))
len(pd.unique(trans_A02.clac_nm3))
len(pd.unique(trans_A02.pd_c))

s = trans_A02.groupby(['clac_nm2']).clnt_age.value_counts(normalize=True).mul(100)
plt.rcParams["figure.figsize"] = (20,15)
s.unstack().plot.barh(stacked=True) ### 연령 비율로 군집화 할 수 없음? ex) 10대가 많이 사는 품목군 (대분류랑 다를듯)

X= s.unstack(level=-1, fill_value=0)

# In general, it's a good idea to scale the data prior to PCA.
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)    
pca = PCA()
x_new = pca.fit_transform(X)
pca.components_[0] # 1주성분
pca.components_[1] # 2주성분

plt.plot(np.cumsum(pca.explained_variance_ratio_))

# kmeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

myplot(x_new[:,0:2],np.transpose(pca.components_[0:2, :]))
plt.show()

plt.rcParams["figure.figsize"] = (10,10)
trans_A02[trans_A02.clnt_age==10].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.clnt_gender=="F")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.clnt_gender=="M")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.weekend==0)].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.weekend==1)].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="F")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="M")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="F")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="M")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)


trans_A02[trans_A02.clnt_age==20].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.clnt_gender=="F")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.clnt_gender=="M")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.weekend==0)].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.weekend==1)].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="F")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="M")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="F")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="M")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)


trans_A02[trans_A02.clnt_age==30].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.clnt_gender=="F")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.clnt_gender=="M")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.weekend==0)].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.weekend==1)].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="F")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="M")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="F")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="M")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)

trans_A02[trans_A02.clnt_age==40].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.clnt_gender=="F")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.clnt_gender=="M")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.weekend==0)].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.weekend==1)].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="F")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="M")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="F")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="M")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)


trans_A02[trans_A02.clnt_age==50].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.clnt_gender=="F")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.clnt_gender=="M")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.weekend==0)].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.weekend==1)].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="F")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="M")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="F")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="M")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)


trans_A02[trans_A02.clnt_age==60].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.clnt_gender=="F")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.clnt_gender=="M")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.weekend==0)].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.weekend==1)].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="F")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="M")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="F")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="M")].clac_nm2.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)


#%% 구체적인 소분류는 뭘 사는지 -3
trans_A02 = trans_info[trans_info.biz_unit == "A02"]
len(pd.unique(trans_A02.clac_nm1))
len(pd.unique(trans_A02.clac_nm3))
len(pd.unique(trans_A02.clac_nm3))


s = trans_A02.groupby(['clac_nm3']).clnt_age.value_counts(normalize=True).mul(100)
plt.rcParams["figure.figsize"] = (20,15)
#s.unstack().plot.barh(stacked=True) ### 연령 비율로 군집화 할 수 없음? ex) 10대가 많이 사는 품목군 (대분류랑 다를듯)

plt.rcParams["figure.figsize"] = (10,10)
trans_A02[trans_A02.clnt_age==10].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.clnt_gender=="F")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.clnt_gender=="M")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.weekend==0)].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.weekend==1)].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="F")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="M")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="F")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==10)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="M")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)


trans_A02[trans_A02.clnt_age==20].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.clnt_gender=="F")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.clnt_gender=="M")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.weekend==0)].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.weekend==1)].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="F")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="M")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="F")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==20)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="M")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)


trans_A02[trans_A02.clnt_age==30].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.clnt_gender=="F")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.clnt_gender=="M")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.weekend==0)].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.weekend==1)].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="F")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="M")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="F")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==30)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="M")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)

trans_A02[trans_A02.clnt_age==40].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.clnt_gender=="F")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.clnt_gender=="M")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.weekend==0)].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.weekend==1)].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="F")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="M")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="F")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==40)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="M")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)


trans_A02[trans_A02.clnt_age==50].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.clnt_gender=="F")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.clnt_gender=="M")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.weekend==0)].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.weekend==1)].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="F")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="M")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="F")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==50)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="M")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)


trans_A02[trans_A02.clnt_age==60].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.clnt_gender=="F")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.clnt_gender=="M")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.weekend==0)].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.weekend==1)].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="F")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.weekend==0)&(trans_A02.clnt_gender=="M")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="F")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)
trans_A02[(trans_A02.clnt_age==60)&(trans_A02.weekend==1)&(trans_A02.clnt_gender=="M")].clac_nm3.value_counts(normalize=True).mul(100).iloc[:10].plot.barh(stacked=True)


#%% 함께 뭘 사는지 (고객별로)
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

dataset = trans_A02[trans_A02.clnt_gender=="F"].groupby(['clnt_id'])['clac_nm1'].apply(list).reset_index()
pd_set = dataset.clac_nm1
t = TransactionEncoder()
t_a = t.fit(pd_set).transform(pd_set)
df = pd.DataFrame(t_a, columns = t.columns_)
df
frequent = apriori(df, min_support=0.01, use_colnames=True)
frequent
association_rules(frequent, metric='confidence', min_threshold=0.5)


dataset = trans_A02[trans_A02.clnt_gender=="M"].groupby(['clnt_id'])['clac_nm1'].apply(list).reset_index()
pd_set = dataset.clac_nm1                    
t = TransactionEncoder()
t_a = t.fit(pd_set).transform(pd_set)
df = pd.DataFrame(t_a, columns = t.columns_)
df
frequent = apriori(df, min_support=0.01, use_colnames=True)
frequent
association_rules(frequent, metric='confidence', min_threshold=0.5)

#%% 성별/연령대별 구매금액, 높은 품목?
dataset = trans_A02[(trans_A02.clnt_gender=="F") & (trans_A02.clnt_age==20) ].groupby(['clnt_id'])['clac_nm1'].apply(list).reset_index()
pd_set = dataset.clac_nm1
t = TransactionEncoder()
t_a = t.fit(pd_set).transform(pd_set)
df = pd.DataFrame(t_a, columns = t.columns_)
df
frequent = apriori(df, min_support=0.01, use_colnames=True)
frequent
association_rules(frequent, metric='confidence', min_threshold=0.5)
