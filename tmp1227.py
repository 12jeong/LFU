%reset -f
#%% import
import os
os.getcwd()
from os import chdir
#os.chdir('C:\\Users\\UOS\\Documents\\GITHUB\\LFU')
os.chdir('C:\\Users\\MYCOM\\Documents\\GITHUB\\LFU')
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

#%% expand output display pd.df
pd.set_option('display.expand_frame_repr', False) 

#%% Load raw data
df_buy  = pd.read_csv("C:\\Users\\MYCOM\\Dropbox\\LFY\\datasets\\df_buy.csv",index_col=0) 
df_nobuy = pd.read_csv("C:\\Users\\MYCOM\\Dropbox\\LFY\\datasets\\df_no_buy.csv",index_col=0) 


#%% buy
df1 = df_buy
df1['hit_diff']=df1.sort_values(['clnt_id','sess_id','sess_dt','hit_seq']).groupby(['clnt_id','sess_id','sess_dt'])['hit_pss_tm'].diff(periods=-1)*-1
df1['hit_diff_ratio'] = df1['hit_diff']/ df1['tot_sess_hr_v'] 

df2 = df1.groupby(['clnt_id','sess_id','sess_dt']).action_type.value_counts().unstack(level=-1, fill_value=0).reset_index()   # count accorinding to action_type
df3 = df1.groupby(['clnt_id','sess_id','sess_dt','action_type'])['hit_diff'].agg('sum').unstack(level=-1, fill_value=0).reset_index()       # time consuming by action_type
df2.rename(columns = {0 : 'action_count_0', 1 : 'action_count_1', 2 : 'action_count_2', 3 : 'action_count_3',
                      4 : 'action_count_4', 5 : 'action_count_5', 6 : 'action_count_6', 7 : 'action_count_7'}, inplace = True)
df3.rename(columns = {0 : 'action_time_0', 1 : 'action_time_1', 2 : 'action_time_2', 3 : 'action_time_3',
                      4 : 'action_time_4', 5 : 'action_time_5', 6 : 'action_time_6', 7 : 'action_time_7'}, inplace = True)
df4 = df1.sort_values(['clnt_id','sess_id','sess_dt','hit_seq']).groupby(['clnt_id','sess_id','sess_dt'])['hit_seq'].agg([('tot_seq','max')])  # total hit_seq
df5 = df1.drop_duplicates(['clnt_id','sess_id','sess_dt']).drop(['hit_seq','action_type','hit_tm','hit_pss_tm','trans_id','sech_kwd','hit_diff','hit_diff_ratio'], axis=1)

df_buy_if = pd.merge(df2,df3,on=['clnt_id','sess_id','sess_dt'], how='inner')
df_buy_if = df_buy_if.merge(df4,on=['clnt_id','sess_id','sess_dt'], how='inner')
df_buy_if = df_buy_if.merge(df5,on=['clnt_id','sess_id','sess_dt'], how='inner')

df_buy_if.head()

#%% nobuy
df_nobuy['buy']= 0 # for Y = 0,1 

df6 = df_nobuy
df6['hit_diff']=df6.sort_values(['clnt_id','sess_id','sess_dt','hit_seq']).groupby(['clnt_id','sess_id','sess_dt'])['hit_pss_tm'].diff(periods=-1)*-1
df6['hit_diff_ratio'] = df6['hit_diff']/ df6['tot_sess_hr_v'] 

df7 = df6.groupby(['clnt_id','sess_id','sess_dt']).action_type.value_counts().unstack(level=-1, fill_value=0).reset_index()   # count accorinding to action_type
df8 = df6.groupby(['clnt_id','sess_id','sess_dt','action_type'])['hit_diff'].agg('sum').unstack(level=-1, fill_value=0).reset_index()       # time consuming by action_type
df7.rename(columns = {0 : 'action_count_0', 1 : 'action_count_1', 2 : 'action_count_2', 3 : 'action_count_3',
                      4 : 'action_count_4', 5 : 'action_count_5', 6 : 'action_count_6', 7 : 'action_count_7'}, inplace = True)
df8.rename(columns = {0 : 'action_time_0', 1 : 'action_time_1', 2 : 'action_time_2', 3 : 'action_time_3',
                      4 : 'action_time_4', 5 : 'action_time_5', 6 : 'action_time_6', 7 : 'action_time_7'}, inplace = True)
df9 = df6.sort_values(['clnt_id','sess_id','sess_dt','hit_seq']).groupby(['clnt_id','sess_id','sess_dt'])['hit_seq'].agg([('tot_seq','max')])  # total hit_seq
df10 = df6.drop_duplicates(['clnt_id','sess_id','sess_dt']).drop(['hit_seq','action_type','hit_tm','hit_pss_tm','trans_id','sech_kwd','hit_diff','hit_diff_ratio'], axis=1)

df_if_tmp = pd.merge(df7,df8,on=['clnt_id','sess_id','sess_dt'], how='inner')
df_if_tmp = df_if_tmp.merge(df9,on=['clnt_id','sess_id','sess_dt'], how='inner')
df_if_tmp = df_if_tmp.merge(df10,on=['clnt_id','sess_id','sess_dt'], how='inner')

#%% final table for customer information
df_design_buy = pd.concat([df_buy_if, df_if_tmp], axis=0)
df_design_buy.to_csv("C:\\Users\\MYCOM\\Dropbox\\LFY\\datasets\\df_design_buy.csv")

#%%
df_design_buy = pd.read_csv("C:\\Users\\MYCOM\\Dropbox\\LFY\\datasets\\df_design_buy.csv",index_col=0)
X = df_design_buy.drop(['sess_dt','sess_id','clnt_id','id','action_count_6','action_count_7','action_time_6','action_time_7'],axis=1)
X = X.dropna()
y = X['buy']
X = X.drop('buy',axis=1)
#X = X[['tot_sess_hr_v']]
X_with_dummies = pd.get_dummies(X,columns=['biz_unit','clnt_age','clnt_gender','dvc_ctg_nm','trfc_src'],drop_first=False)

(X == 'unknown').sum()
(y == 1).sum()
(y == 0).sum()
from sklearn import tree
buy_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
#buy_tree.fit(X_with_dummies, y)
buy_tree.fit(X_with_dummies, y)

from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image

#import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

dot_data = export_graphviz(buy_tree, out_file=None, class_names=["nobuy", "buy"],
                           feature_names=list(X_with_dummies.columns) , impurity=False, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
