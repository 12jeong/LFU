%reset -f
#%%
# -- IMPORT PACKAGE
import os
os.getcwd()
from os import chdir
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sklearn
import matplotlib as mpl

# -- DIRECTORY
pcsh = "C:\\Users\\user\\"
pc = pcsh
os.chdir(pcsh)
pdir = os.getcwd() ; print(pdir) #os.chdir('C:\\Users\\UOS\\Documents\\GITHUB\LFU')
# -- PLOT STYLE
plt.style.use('ggplot') # 그래프의 스타일을 지정
mpl.rcParams.update({'font.size':14})
plt.rc('font',family='Malgun Gothic') # windows
%matplotlib inline 
#%% expand output display pd.df
pd.set_option('display.expand_frame_repr', False) 

#%% Load raw data
df_buy  = pd.read_csv(pcsh+"Dropbox\\LFY\\datasets\\ppdata\\df_buy.csv") 
df_nobuy = pd.read_csv(pcsh+"Dropbox\\LFY\\datasets\\ppdata\\df_nobuy.csv")
online_bh = pd.read_csv(pcsh+"Dropbox\\LFY\\datasets\\ppdata\\online_bh.csv")
trans_info = pd.read_csv(pcsh+"Dropbox\\LFY\\datasets\\ppdata\\trans_info.csv")
df_design_buy = pd.read_csv(pcsh+"Dropbox\\LFY\\datasets\\ppdata\\df_design_buy.csv",index_col=0)
mg_ppdata = pd.read_csv(pcsh+"Dropbox\\LFY\\datasets\\ppdata\\mg_ppdata.csv")
#%%
len(np.unique(trans_info[trans_info.biz_unit=="B03"].clac_nm1))
len(np.unique(trans_info[trans_info.biz_unit=="B03"].clac_nm2))
len(np.unique(trans_info[trans_info.biz_unit=="B03"].clac_nm3))
len(np.unique(trans_info[trans_info.biz_unit=="B03"].pd_c))

#%% biz_unit에 따라 성별/연령/요일 별로 많이 사는 품목 EDA (그림은 Percentage(%)로 그려짐)
def EDA_clac1_by_biz(biz_unit = None, clnt_age = None, clnt_gender = None, weekend = None):
    trans_tmp = trans_info[trans_info.biz_unit == biz_unit]
    if ( pd.isnull(clnt_age) & pd.isnull(clnt_gender) & pd.isnull(weekend)):
        clac_counts = trans_tmp.clac_nm1.value_counts(normalize=True).mul(100)
    if ( pd.isnull(clnt_age) & pd.notnull(clnt_gender) & pd.isnull(weekend)):
        clac_counts = trans_tmp[trans_tmp.clnt_gender==clnt_gender].clac_nm1.value_counts(normalize=True).mul(100)
    if ( pd.notnull(clnt_age) & pd.isnull(clnt_gender) & pd.isnull(weekend)):
        clac_counts = trans_tmp[trans_tmp.clnt_age==clnt_age].clac_nm1.value_counts(normalize=True).mul(100)
    if ( pd.notnull(clnt_age) & pd.notnull(clnt_gender) & pd.isnull(weekend)):
        clac_counts = trans_tmp[(trans_tmp.clnt_age==clnt_age)&(trans_tmp.clnt_gender==clnt_gender)].clac_nm1.value_counts(normalize=True).mul(100)
    if ( pd.notnull(clnt_age) & pd.notnull(clnt_gender) & pd.notnull(weekend)):
        clac_counts = trans_tmp[(trans_tmp.clnt_age==clnt_age)&(trans_tmp.weekend==weekend)&(trans_tmp.clnt_gender==clnt_gender)].clac_nm1.value_counts(normalize=True).mul(100)
    plt.rcParams["figure.figsize"] = (10,10)
    clac_counts.iloc[:5].plot.barh(stacked=True)
    plt.show()    
    return(clac_counts.iloc[:3])

# a = pd.DataFrame()
# for x in [10,20,30,40,50,60]:
#     a_tmp = EDA_clac1_by_biz(biz_unit="A01", clnt_age=x).reset_index()    
#     if (a_tmp.shape[0]==1):
#         a_tmp = pd.DataFrame([EDA_clac1_by_biz(biz_unit="A01", clnt_age=x).index[0],'empty','empty'], columns=['index'])
#     a = a.append(a_tmp['index'].tolist()) 
    

def EDA_clac2_by_biz(biz_unit = None, clnt_age = None, clnt_gender = None, weekend = None):
    trans_tmp = trans_info[trans_info.biz_unit == biz_unit]
    if ( pd.isnull(clnt_age) & pd.isnull(clnt_gender) & pd.isnull(weekend)):
        clac_counts = trans_tmp.clac_nm2.value_counts(normalize=True).mul(100)
    if ( pd.isnull(clnt_age) & pd.notnull(clnt_gender) & pd.isnull(weekend)):
        clac_counts = trans_tmp[trans_tmp.clnt_gender==clnt_gender].clac_nm2.value_counts(normalize=True).mul(100)
    if ( pd.notnull(clnt_age) & pd.isnull(clnt_gender) & pd.isnull(weekend)):
        clac_counts = trans_tmp[trans_tmp.clnt_age==clnt_age].clac_nm2.value_counts(normalize=True).mul(100)
    if ( pd.notnull(clnt_age) & pd.notnull(clnt_gender) & pd.isnull(weekend)):
        clac_counts = trans_tmp[(trans_tmp.clnt_age==clnt_age)&(trans_tmp.clnt_gender==clnt_gender)].clac_nm2.value_counts(normalize=True).mul(100)
    if ( pd.notnull(clnt_age) & pd.notnull(clnt_gender) & pd.notnull(weekend)):
        clac_counts = trans_tmp[(trans_tmp.clnt_age==clnt_age)&(trans_tmp.weekend==weekend)&(trans_tmp.clnt_gender==clnt_gender)].clac_nm2.value_counts(normalize=True).mul(100)
    plt.rcParams["figure.figsize"] = (10,10)
    clac_counts.iloc[:5].plot.barh(stacked=True)
    plt.show()    
    return(clac_counts.iloc[:3])

def EDA_clac3_by_biz(biz_unit = None, clnt_age = None, clnt_gender = None, weekend = None):
    trans_tmp = trans_info[trans_info.biz_unit == biz_unit]
    if ( pd.isnull(clnt_age) & pd.isnull(clnt_gender) & pd.isnull(weekend)):
        clac_counts = trans_tmp.clac_nm3.value_counts(normalize=True).mul(100)
    if ( pd.isnull(clnt_age) & pd.notnull(clnt_gender) & pd.isnull(weekend)):
        clac_counts = trans_tmp[trans_tmp.clnt_gender==clnt_gender].clac_nm3.value_counts(normalize=True).mul(100)
    if ( pd.notnull(clnt_age) & pd.isnull(clnt_gender) & pd.isnull(weekend)):
        clac_counts = trans_tmp[trans_tmp.clnt_age==clnt_age].clac_nm3.value_counts(normalize=True).mul(100)
    if ( pd.notnull(clnt_age) & pd.notnull(clnt_gender) & pd.isnull(weekend)):
        clac_counts = trans_tmp[(trans_tmp.clnt_age==clnt_age)&(trans_tmp.clnt_gender==clnt_gender)].clac_nm3.value_counts(normalize=True).mul(100)
    if ( pd.notnull(clnt_age) & pd.notnull(clnt_gender) & pd.notnull(weekend)):
        clac_counts = trans_tmp[(trans_tmp.clnt_age==clnt_age)&(trans_tmp.weekend==weekend)&(trans_tmp.clnt_gender==clnt_gender)].clac_nm3.value_counts(normalize=True).mul(100)
    plt.rcParams["figure.figsize"] = (10,10)
    clac_counts.iloc[:5].plot.barh(stacked=True)
    plt.show()    
    return(clac_counts.iloc[:3])

#%% 
biz_unit = "B03"

trans_info[(trans_info.biz_unit=="B03")&(trans_info.clnt_age==60)].shape

EDA_clac1_by_biz(biz_unit=biz_unit)
EDA_clac1_by_biz(biz_unit=biz_unit, clnt_age=10)
EDA_clac1_by_biz(biz_unit=biz_unit, clnt_age=20)
EDA_clac1_by_biz(biz_unit=biz_unit, clnt_age=30)
EDA_clac1_by_biz(biz_unit=biz_unit, clnt_age=40)
EDA_clac1_by_biz(biz_unit=biz_unit, clnt_age=50)
EDA_clac1_by_biz(biz_unit=biz_unit, clnt_age=60)
EDA_clac1_by_biz(biz_unit=biz_unit, clnt_gender="F")
EDA_clac1_by_biz(biz_unit=biz_unit, clnt_gender="M")
EDA_clac1_by_biz(biz_unit=biz_unit, clnt_age=10, clnt_gender="F")
EDA_clac1_by_biz(biz_unit=biz_unit, clnt_age=10, clnt_gender="M")
EDA_clac1_by_biz(biz_unit=biz_unit, clnt_age=20, clnt_gender="F")
EDA_clac1_by_biz(biz_unit=biz_unit, clnt_age=20, clnt_gender="M")
EDA_clac1_by_biz(biz_unit=biz_unit, clnt_age=30, clnt_gender="F")
EDA_clac1_by_biz(biz_unit=biz_unit, clnt_age=30, clnt_gender="M")
EDA_clac1_by_biz(biz_unit=biz_unit, clnt_age=40, clnt_gender="F")
EDA_clac1_by_biz(biz_unit=biz_unit, clnt_age=40, clnt_gender="M")
EDA_clac1_by_biz(biz_unit=biz_unit, clnt_age=50, clnt_gender="F")
EDA_clac1_by_biz(biz_unit=biz_unit, clnt_age=50, clnt_gender="M")
EDA_clac1_by_biz(biz_unit=biz_unit, clnt_age=60, clnt_gender="F")
EDA_clac1_by_biz(biz_unit=biz_unit, clnt_age=60, clnt_gender="M")


#%% 
biz_unit = "B03"
EDA_clac3_by_biz(biz_unit=biz_unit)
EDA_clac3_by_biz(biz_unit=biz_unit, clnt_age=10)
EDA_clac3_by_biz(biz_unit=biz_unit, clnt_age=20)
EDA_clac3_by_biz(biz_unit=biz_unit, clnt_age=30)
EDA_clac3_by_biz(biz_unit=biz_unit, clnt_age=40)
EDA_clac3_by_biz(biz_unit=biz_unit, clnt_age=50)
EDA_clac3_by_biz(biz_unit=biz_unit, clnt_age=60)
EDA_clac3_by_biz(biz_unit=biz_unit, clnt_gender="F")
EDA_clac3_by_biz(biz_unit=biz_unit, clnt_gender="M")
EDA_clac3_by_biz(biz_unit=biz_unit, clnt_age=10, clnt_gender="F")
EDA_clac3_by_biz(biz_unit=biz_unit, clnt_age=10, clnt_gender="M")
EDA_clac3_by_biz(biz_unit=biz_unit, clnt_age=20, clnt_gender="F")
EDA_clac3_by_biz(biz_unit=biz_unit, clnt_age=20, clnt_gender="M")
EDA_clac3_by_biz(biz_unit=biz_unit, clnt_age=30, clnt_gender="F")
EDA_clac3_by_biz(biz_unit=biz_unit, clnt_age=30, clnt_gender="M")
EDA_clac3_by_biz(biz_unit=biz_unit, clnt_age=40, clnt_gender="F")
EDA_clac3_by_biz(biz_unit=biz_unit, clnt_age=40, clnt_gender="M")
EDA_clac3_by_biz(biz_unit=biz_unit, clnt_age=50, clnt_gender="F")
EDA_clac3_by_biz(biz_unit=biz_unit, clnt_age=50, clnt_gender="M")
EDA_clac3_by_biz(biz_unit=biz_unit, clnt_age=60, clnt_gender="F")
EDA_clac3_by_biz(biz_unit=biz_unit, clnt_age=60, clnt_gender="M")


#%% 마트데이터 손님 군집화 (구매 상품-clac_nm2을 중심으로)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 상품의 단가 구하기 = buy_am/buy_ct
# trans_info['buy_am_unit'] = trans_info['buy_am']/trans_info['buy_ct']
trans_info['buy_am_new'] = trans_info['buy_am']/trans_info['buy_ct']
# 상품분류별로 표준화하기 
# trans_info['buy_am_std'] = trans_info.groupby('clac_nm2')['buy_am_unit'].apply(lambda x: (x - x.mean()) / x.std())
# 표준화 한 단가에 buy_ct 곱해서 구매지수? 만들기
# trans_info['buy_am_new'] = trans_info.buy_am_std * trans_info.buy_ct
# 마트데이터만 추출 
df_mart_tmp = trans_info[(trans_info.biz_unit == "A03")|(trans_info.biz_unit == "B01")|(trans_info.biz_unit == "B02")]
df_mart_tmp2 = df_mart_tmp[['clnt_id','clac_nm2','buy_am_new']].groupby(['clnt_id','clac_nm2'])['buy_am_new'].agg('sum')
df_mart = df_mart_tmp2.unstack(level=-1, fill_value=0)


X = df_mart
X_clnt = df_mart.index
X_clac = df_mart.columns

# In general, it's a good idea to scale the data prior to PCA.
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)    
pca = PCA()
x_new = pca.fit_transform(X)
pca.components_[0] # 1주성분
pca.components_[1] # 2주성분

pd.DataFrame([X_clac,pca.components_[0],pca.components_[1]]).transpose()
plt.plot(np.cumsum(pca.explained_variance_ratio_))

# kmeans
kmeans = KMeans(n_clusters=10)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# plot PCA loading and loading in biplot
y = y_kmeans
def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = y)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

myplot(x_new[:,0:2],np.transpose(pca.components_[0:2, :]))
plt.show()


# network by corrleation matrix
import networkx as nx

def NETcorr(dataframe,BIZ_set,ID,CL,SCORE,threshold,title):
    
    df_temp = dataframe[dataframe.biz_unit.isin(BIZ_set)]
    df_temp2 = df_temp[[ID,CL,SCORE]].groupby([ID,CL])[SCORE].agg('sum')
    df_temp3 = df_temp2.unstack(level=-1, fill_value=0)

    corr = df_temp3.corr()
    corr.columns.name = None
    links = corr.stack().reset_index()
    links.columns = ['var1', 'var2','value']
    links_filtered=links.loc[ (links['value'] >threshold) & (links['var1'] != links['var2']) ]
    G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')
    nx.draw(G, with_labels=True, node_color='orange', node_size=50, edge_color='black', linewidths=1, font_size=10)

    plt.title(title)
    plt.gcf().canvas.set_window_title("")
    plt.show()

threshold = 0.4
NETcorr(trans_info,["A03","B01","B02"],"clnt_id","clac_nm2","buy_am",threshold,"Mart") # Mart
NETcorr(trans_info,["A01"],"clnt_id","clac_nm2","buy_am",threshold,"Department store") # Department store
NETcorr(trans_info, ["A02"],"clnt_id","clac_nm2","buy_am",threshold,"Outlet") # Outlet
NETcorr(trans_info,["B03"],"clnt_id","clac_nm2","buy_am",threshold,"cosmetics") # cosmetics
