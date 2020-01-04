%reset -f
#%% import
import os
os.getcwd()
from os import chdir
os.chdir('C:\\Users\\UOS\\Dropbox')
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

import shelve
#import pickle # for save.image like R 

# 장바구니분석
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


#%% expand output display pd.df
pd.set_option('display.expand_frame_repr', False) 

#%% Load raw data
online_bh  = pd.read_csv(".\LFY\\datasets\\ppdata\\online_bh.csv",index_col=0) 
trans_info = pd.read_csv(".\LFY\\datasets\\ppdata\\trans_info.csv",index_col=0) 

mg_ppdata = pd.read_csv(".\LFY\\datasets\\ppdata\\mg_ppdata.csv",index_col=0) 

#%% 각 사이트에 주된 상품군은 무엇인가?
mg_A01 = mg_ppdata[mg_ppdata.biz_unit=="A01"][['biz_unit','kwd_list','clac_nm1','clac_nm2','clac_nm3']]
mg_A02 = mg_ppdata[mg_ppdata.biz_unit=="A02"][['biz_unit','kwd_list','clac_nm1','clac_nm2','clac_nm3']]
mg_A03 = mg_ppdata[mg_ppdata.biz_unit=="A03"][['biz_unit','kwd_list','clac_nm1','clac_nm2','clac_nm3']]

pd.isnull(mg_A01.clac_nm1).sum()
pd.isnull(mg_A02.clac_nm1).sum()
pd.isnull(mg_A03.clac_nm1).sum()

# A01
clacs =  mg_A01.clac_nm1[~pd.isnull(mg_A01.clac_nm1)].apply(lambda x : x.replace(" ", ""))
list_A01 = []
for i in clacs :
    for g in i.split(',') :
        list_A01.append(g)
clac_unique = pd.unique(list_A01)
zero = np.zeros(((len(clacs),len(clac_unique))))
dummy = pd.DataFrame(zero, columns =clac_unique)
for n, g in enumerate(clacs):   
    dummy.ix[n, g.split(",")]=1         
TDM = dummy.T
print(TDM)

clac_counter = TDM.sum(axis=1)  
clac_counter.sort_values(ascending=False).iloc[:10].plot(kind='barh',title='A01 clac counter')

# A02
clacs =  mg_A02.clac_nm1[~pd.isnull(mg_A02.clac_nm1)].apply(lambda x : x.replace(" ", ""))
list_A02 = []
for i in clacs :
    for g in i.split(',') :
        list_A02.append(g)
clac_unique = pd.unique(list_A02)
zero = np.zeros(((len(clacs),len(clac_unique))))
dummy = pd.DataFrame(zero, columns =clac_unique)
for n, g in enumerate(clacs):   
    dummy.ix[n, g.split(",")]=1         
TDM = dummy.T
print(TDM)

clac_counter = TDM.sum(axis=1)  
clac_counter.sort_values(ascending=False).iloc[:10].plot(kind='barh',title='A02 clac counter')

# A03
clacs =  mg_A03.clac_nm1[~pd.isnull(mg_A03.clac_nm1)].apply(lambda x : x.replace(" ", ""))
list_A03 = []
for i in clacs :
    for g in i.split(',') :
        list_A03.append(g)
clac_unique = pd.unique(list_A03)
zero = np.zeros(((len(clacs),len(clac_unique))))
dummy = pd.DataFrame(zero, columns =clac_unique)
for n, g in enumerate(clacs):   
    dummy.ix[n, g.split(",")]=1         
TDM = dummy.T
print(TDM)

clac_counter = TDM.sum(axis=1)  
clac_counter.sort_values(ascending=False).iloc[:10].plot(kind='barh',title='A03 clac counter')


#%% 장바구니분석을 해보고싶어 (A01, A02, A03의 대분류로 알아보자)
#dataset = mg_ppdata
dataset = mg_ppdata[mg_ppdata.biz_unit=="A02"]
clac_set = dataset.clac_nm1[~pd.isnull(dataset.clac_nm1)].apply(lambda x : x.replace(" " ,"")).apply(lambda x : x.split(",")).array

t = TransactionEncoder()
t_a = t.fit(clac_set).transform(clac_set)
df = pd.DataFrame(t_a, columns = t.columns_)
df
frequent = apriori(df, min_support=0.001, use_colnames=True)
frequent

association_rules(frequent, metric='confidence', min_threshold=0.01)


#%% A03 + 대분류
len(pd.unique(trans_info.clac_nm1)) # 60개

dataset = trans_info.groupby(['biz_unit','clnt_id','de_dt','de_tm','trans_id'])['clac_nm1'].apply(list).reset_index()
pd_set = dataset.clac_nm1[dataset.biz_unit=="A03"]

t = TransactionEncoder()
t_a = t.fit(pd_set).transform(pd_set)
df = pd.DataFrame(t_a, columns = t.columns_)
df
frequent = apriori(df, min_support=0.2, use_colnames=True)
frequent

association_rules(frequent, metric='confidence', min_threshold=0.2)

#%% A03 + 중분류
len(pd.unique(trans_info.clac_nm2)) # 60개

dataset = trans_info.groupby(['biz_unit','clnt_id','de_dt','de_tm','trans_id'])['clac_nm2'].apply(list).reset_index()
pd_set = dataset.clac_nm2[dataset.biz_unit=="A03"]

t = TransactionEncoder()
t_a = t.fit(pd_set).transform(pd_set)
df = pd.DataFrame(t_a, columns = t.columns_)
df
frequent = apriori(df, min_support=0.1, use_colnames=True)
frequent

association_rules(frequent, metric='confidence', min_threshold=0.1)


#%% B01 + 대분류
dataset = trans_info.groupby(['biz_unit','clnt_id','de_dt','de_tm','trans_id'])['clac_nm1'].apply(list).reset_index()
pd_set = dataset.clac_nm1[dataset.biz_unit=="B01"]

t = TransactionEncoder()
t_a = t.fit(pd_set).transform(pd_set)
df = pd.DataFrame(t_a, columns = t.columns_)
df
frequent = apriori(df, min_support=0.1, use_colnames=True)
frequent

association_rules(frequent, metric='confidence', min_threshold=0.1)

#%% B02 + 대분류
dataset = trans_info.groupby(['biz_unit','clnt_id','de_dt','de_tm','trans_id'])['clac_nm1'].apply(list).reset_index()
pd_set = dataset.clac_nm1[dataset.biz_unit=="B02"]

t = TransactionEncoder()
t_a = t.fit(pd_set).transform(pd_set)
df = pd.DataFrame(t_a, columns = t.columns_)
df
frequent = apriori(df, min_support=0.1, use_colnames=True)
frequent

association_rules(frequent, metric='confidence', min_threshold=0.1)

#%% B03 + 대분류
dataset = trans_info.groupby(['biz_unit','clnt_id','de_dt','de_tm','trans_id'])['clac_nm1'].apply(list).reset_index()
pd_set = dataset.clac_nm1[dataset.biz_unit=="B03"]

t = TransactionEncoder()
t_a = t.fit(pd_set).transform(pd_set)
df = pd.DataFrame(t_a, columns = t.columns_)
df
frequent = apriori(df, min_support=0.1, use_colnames=True)
frequent

association_rules(frequent, metric='confidence', min_threshold=0.1)


#%% 장바구니 분석을 해보고 싶어 22 (pd_c로 알아보자)
len(pd.unique(trans_info.pd_c)) # 1664개 밖에 안됨

dataset = trans_info.groupby(['biz_unit','clnt_id','de_dt','de_tm','trans_id'])['pd_c'].apply(list).reset_index()
pd_set = dataset.pd_c[dataset.biz_unit=="A03"]

t = TransactionEncoder()
t_a = t.fit(pd_set).transform(pd_set)
df = pd.DataFrame(t_a, columns = t.columns_)
df
frequent = apriori(df, min_support=0.05, use_colnames=True)
frequent

association_rules(frequent, metric='confidence', min_threshold=0.1)

trans_info[trans_info.pd_c==347][['pd_c','clac_nm3']].drop_duplicates()
trans_info[trans_info.pd_c==964][['pd_c','clac_nm3']].drop_duplicates()
trans_info[trans_info.pd_c==1395][['pd_c','clac_nm3']].drop_duplicates()
trans_info[trans_info.pd_c==1617][['pd_c','clac_nm3']].drop_duplicates()