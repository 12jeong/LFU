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
os.chdir('C:/Users/MYCOM/Dropbox')
pdir = os.getcwd() ;print(pdir)

# -- PLOT STYLE
plt.style.use('ggplot') # 그래프의 스타일을 지정
mpl.rcParams.update({'font.size':14})
plt.rc('font',family='Malgun Gothic') # windows
%matplotlib inline 
#%% expand output display pd.df
pd.set_option('display.expand_frame_repr', False) 

#%% Load data
df_buy  = pd.read_csv(".\LFY/datasets/ppdata/df_buy.csv") 
online_bh  = pd.read_csv(".\LFY/datasets/ppdata/online_bh.csv" )
trans_info = pd.read_csv(".\LFY/datasets/ppdata/trans_info.csv") 
# df_nobuy = pd.read_csv("./LFY/datasets/ppdata/df_nobuy.csv") 

online_bh['sess_dt'] =  pd.to_datetime(online_bh['sess_dt'])
trans_info['de_dt'] =  pd.to_datetime(trans_info['de_dt'])
df_buy['sess_dt'] =  pd.to_datetime(df_buy['sess_dt'])

#%% create variable by clnt
# -- clnt별 접속 세션 수
clnt_login_tmp = online_bh.drop_duplicates(['biz_unit','clnt_id','sess_dt','sess_id'])
clnt_login_tmp2 = clnt_login_tmp.groupby(['biz_unit','clnt_id','sess_dt'])['sess_id'].agg({'login_num':'size'}).reset_index()
clnt_login = clnt_login_tmp2.groupby(['biz_unit','clnt_id'])['login_num'].agg('sum').reset_index()

# -- clnt별 구매 세션 수
clnt_buy_tmp = df_buy.drop_duplicates(['biz_unit','clnt_id','sess_dt','sess_id'])
clnt_buy_tmp2 = clnt_buy_tmp.groupby(['biz_unit','clnt_id','sess_dt'])['sess_id'].agg({'buy_num':'size'}).reset_index()
clnt_buy = clnt_buy_tmp2.groupby(['biz_unit','clnt_id'])['buy_num'].agg('sum').reset_index()


# -- clnt별 접속주기 (일) - 단기, 장기
clnt_interval_tmp = online_bh.drop_duplicates(['biz_unit','clnt_id','sess_dt'])
clnt_interval_tmp['days_diff'] = clnt_interval_tmp.sort_values(['biz_unit','clnt_id','sess_dt']).groupby(['biz_unit','clnt_id'])['sess_dt'].diff()*-1
# ttmp = clnt_interval_tmp.sort_values(['biz_unit','clnt_id','sess_dt'])[['biz_unit','clnt_id','sess_dt','days_diff']]
clnt_interval_tmp['days_diff'] = clnt_interval_tmp['days_diff'].dt.days*-1
# sns.catplot(y="days_diff", kind="box", data=clnt_interval_tmp)
# clnt_interval_tmp['days_diff'][clnt_interval_tmp['days_diff'].isna()] = 
# 처음방문(0)이 아닌 재방문 고객중 접속주기가 3일 이하인 개수
clnt_interval_tmp['days_le3'] = 0
clnt_interval_tmp['days_le3'][(clnt_interval_tmp['days_diff'] > 0) & (clnt_interval_tmp['days_diff'] <= 3)] = 1
# 처음방문(0)이 아닌 재방문 고객중 접속주기가 10일 이하인 개수
clnt_interval_tmp['days_le10'] = 0
clnt_interval_tmp['days_le10'][(clnt_interval_tmp['days_diff'] > 0) & (clnt_interval_tmp['days_diff'] > 3) & (clnt_interval_tmp['days_diff'] <= 10)] = 1
# 처음방문(0)이 아닌 재방문 고객중 접속주기가 10일 초과인 개수
clnt_interval_tmp['days_gt10'] = 0
clnt_interval_tmp['days_gt10'][clnt_interval_tmp['days_diff'] > 10] = 1
# ttmp = clnt_interval_tmp.sort_values(['biz_unit','clnt_id','sess_dt'])[['biz_unit','clnt_id','sess_dt','days_diff','days_in5','days_in30']] 
clnt_interval_tmp2 = clnt_interval_tmp.groupby(['biz_unit','clnt_id']).agg({'days_le3':'sum'}).reset_index()
clnt_interval_tmp3 = clnt_interval_tmp.groupby(['biz_unit','clnt_id']).agg({'days_le10':'sum'}).reset_index()
clnt_interval_tmp4 = clnt_interval_tmp.groupby(['biz_unit','clnt_id']).agg({'days_gt10':'sum'}).reset_index()
clnt_interval = clnt_interval_tmp[['clnt_id','biz_unit']].merge(clnt_interval_tmp2, how="inner").merge(clnt_interval_tmp3, how="inner").merge(clnt_interval_tmp4, how="inner")
# clnt_interval_tmp['days_diff'].median()
# clnt_interval_tmp['days_diff'].describe()
# np.quantile(clnt_interval_tmp[~pd.isnull(clnt_interval_tmp.days_diff)].days_diff, 0.5)
# np.quantile(clnt_interval_tmp[~pd.isnull(clnt_interval_tmp.days_diff)].days_diff, 0.8)

# -- clnt별 구입주기 (일) - 단기, 장기
buy_interval_tmp = df_buy.drop_duplicates(['biz_unit','clnt_id','sess_dt'])
buy_interval_tmp['buy_diff'] = buy_interval_tmp.sort_values(['biz_unit','clnt_id','sess_dt']).groupby(['biz_unit','clnt_id'])['sess_dt'].diff()*-1
# ttmp = buy_interval_tmp.sort_values(['biz_unit','clnt_id','sess_dt'])[['biz_unit','clnt_id','sess_dt','buy_diff']]
buy_interval_tmp['buy_diff'] = buy_interval_tmp['buy_diff'].dt.days*-1
# sns.catplot(y="buy_diff", kind="box", data=buy_interval_tmp)
# buy_interval_tmp['buy_diff'][buy_interval_tmp['buy_diff'].isna()] = 
# 처음방문(0)이 아닌 재방문 고객중 구매주기가 7일 이하인 개수
buy_interval_tmp['buy_le7'] = 0
buy_interval_tmp['buy_le7'][(buy_interval_tmp['buy_diff'] > 0) & (buy_interval_tmp['buy_diff'] <= 10)] = 1
# 처음방문(0)이 아닌 재방문 고객중 구매주기가 20일 이하인 개수
buy_interval_tmp['buy_le20'] = 0
buy_interval_tmp['buy_le20'][(buy_interval_tmp['buy_diff'] > 0) & (buy_interval_tmp['buy_diff'] > 10) & (buy_interval_tmp['buy_diff'] <= 20)] = 1
# 처음방문(0)이 아닌 재방문 고객중 구매주기가 20일 초과인 개수
buy_interval_tmp['buy_gt20'] = 0
buy_interval_tmp['buy_gt20'][buy_interval_tmp['buy_diff'] > 20] = 1

# ttmp = buy_interval_tmp.sort_values(['biz_unit','clnt_id','sess_dt'])[['biz_unit','clnt_id','sess_dt','buy_diff','days_in5','days_in30']] 
buy_interval_tmp2 = buy_interval_tmp.groupby(['biz_unit','clnt_id']).agg({'buy_le7':'sum'}).reset_index()
buy_interval_tmp3 = buy_interval_tmp.groupby(['biz_unit','clnt_id']).agg({'buy_le20':'sum'}).reset_index()
buy_interval_tmp4 = buy_interval_tmp.groupby(['biz_unit','clnt_id']).agg({'buy_gt20':'sum'}).reset_index()
buy_interval = pd.merge(buy_interval_tmp2, buy_interval_tmp3, how="inner").merge(buy_interval_tmp4, how="inner")

# buy_interval_tmp['buy_diff'].median()
# buy_interval_tmp['buy_diff'].describe()
# np.quantile(buy_interval_tmp[~pd.isnull(buy_interval_tmp.buy_diff)].buy_diff, 0.5)
# np.quantile(buy_interval_tmp[~pd.isnull(buy_interval_tmp.buy_diff)].buy_diff, 0.8)

# -- merge for clnt_info
df_tmp = online_bh.drop_duplicates(['biz_unit','clnt_id'])[['biz_unit','clnt_id','tot_pag_view_ct','tot_sess_hr_v','trfc_src','dvc_ctg_nm','clnt_gender','clnt_age']]
# -- clnt 가입정보가 있는지 없는지 (회원/비회원)
df_tmp['member'] = 1
df_tmp['member'][df_tmp.clnt_gender == "unknown"]= 0

clnt_info_tmp = pd.merge(df_tmp,clnt_login, how="left").merge(clnt_buy,how="left").merge(clnt_interval,how="left").merge(buy_interval,how="left" )



#%% create variable by session
df1 = online_bh
# -- 세션 별로 각 action에 소모한 시간
df1['hit_diff']=df1.sort_values(['biz_unit','clnt_id','sess_id','sess_dt','hit_seq']).groupby(['biz_unit','clnt_id','sess_id','sess_dt'])['hit_pss_tm'].diff(periods=-1)*-1
df1['hit_diff_ratio'] = df1['hit_diff']/ df1['tot_sess_hr_v'] # 시간의 비율
# -- 세션 내에서 검색한 unique한 검색어 개수
df_kwd_tmp = df1[~pd.isnull(df1.sech_kwd)].groupby(['biz_unit','clnt_id','sess_id','sess_dt'])['sech_kwd'].apply(lambda x: len(np.unique(list(x)))).reset_index()
df_kwd = df_kwd_tmp.groupby(['biz_unit','clnt_id'])['sech_kwd'].agg('mean').reset_index()
# -- 세션 내에서 가장 관심이 오래 있었던 제품의 세부정보 보기 시간
df_max_2time_tmp = df1[df1.action_type==2].groupby(['biz_unit','clnt_id','sess_id','sess_dt'])['hit_diff'].agg({'max_2time':'max'})
df_max_2time = df_max_2time_tmp.groupby(['biz_unit','clnt_id'])['max_2time'].agg('mean').reset_index()
# -- 첫 구매까지 걸린 시각 
df_buytime_tmp = df1[df1.action_type==6].sort_values(['biz_unit','clnt_id','sess_id','sess_dt','hit_seq']
                                                 ).groupby(['biz_unit','clnt_id','sess_id','sess_dt'])['hit_pss_tm'].nth(0).reset_index()   
df_buytime = df_buytime_tmp.groupby(['biz_unit','clnt_id'])['hit_pss_tm'].agg('mean').reset_index()
df_buytime.columns = ['biz_unit','clnt_id','buy_time']
# 각 행동 수행 빈도 (freq)
total_hit_seq = df1.sort_values(['biz_unit','clnt_id','sess_id','sess_dt','hit_seq']).groupby(['biz_unit','clnt_id','sess_id','sess_dt']).nth(-1)['hit_seq']
count_hit_seq_tmp = df1.groupby(['biz_unit','clnt_id','sess_id','sess_dt','action_type'])['hit_seq'].agg('count')
freq_hit_seq_tmp = count_hit_seq_tmp/total_hit_seq
df2 = freq_hit_seq_tmp.unstack(level=-1, fill_value= 'NaN' ).reset_index()   # count accorinding to action_type
df2.rename(columns = {0 : 'action_count_0', 1 : 'action_count_1', 2 : 'action_count_2', 3 : 'action_count_3',
                      4 : 'action_count_4', 5 : 'action_count_5', 6 : 'action_count_6', 7 : 'action_count_7'}, inplace = True)
df_action_freq = df2.groupby(['biz_unit','clnt_id']).agg('mean').reset_index().drop('sess_id',axis=1)

# -- 각 행동 소요시간 비율 (ratio)
df3 = df1.groupby(['biz_unit','clnt_id','sess_id','sess_dt','action_type'])['hit_diff_ratio'].agg('sum').unstack(level=-1, fill_value='NaN').reset_index()       # time consuming by action_type
df3.rename(columns = {0 : 'action_time_0', 1 : 'action_time_1', 2 : 'action_time_2', 3 : 'action_time_3',
                      4 : 'action_time_4', 5 : 'action_time_5', 6 : 'action_time_6', 7 : 'action_time_7'}, inplace = True)
df_action_time = df3.groupby(['biz_unit','clnt_id']).agg('mean').reset_index().drop('sess_id',axis=1)

# -- merge for clnt_info
clnt_info = clnt_info_tmp.merge(df_kwd,how="left").merge(df_max_2time,how="left").merge(df_buytime,how="left").merge(df_action_freq,how="left").merge(df_action_time,how="left")
clnt_info.columns
clnt_info = clnt_info.drop(['clnt_gender','action_count_4','action_count_5','action_count_6','action_count_7',
                            'action_time_4','action_time_5','action_time_6','action_time_7'],axis=1)
clnt_info['dvc_ctg_nm'][clnt_info.dvc_ctg_nm=='mobile_app'] = 'mobile'
clnt_info['dvc_ctg_nm'][clnt_info.dvc_ctg_nm=='mobile_web'] = 'mobile'
clnt_info.columns
#%% final table for customer information
clnt_info.to_csv("./LFY/datasets/ppdata/clnt_info.csv",index=False)

#%% K-means
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# clnt_info = pd.read_csv("./LFY/datasets/ppdata/clnt_info.csv")
# only for buy client
clnt_buy_info = clnt_info[~pd.isnull(clnt_info.buy_num)]
clnt_buy_A01 = clnt_buy_info [clnt_buy_info.biz_unit == "A01"]
clnt_buy_A02 = clnt_buy_info [clnt_buy_info.biz_unit == "A02"]
clnt_buy_A03 = clnt_buy_info [clnt_buy_info.biz_unit == "A03"]
# continuos variable for kmeans
X = clnt_buy_A01.drop(['biz_unit','clnt_id','trfc_src','dvc_ctg_nm','clnt_age','member'],axis=1)
X_col = X.columns
X = X.dropna()
# In general, it's a good idea to scale the data 
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)    

# kmeans
kmeans = KMeans(n_clusters=10)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

X_df = pd.DataFrame(X)
X_df['y'] = y_kmeans

for i in range(19):
   fig, ax = plt.subplots(figsize=(10,3))
   plt.suptitle('')
   X_df.boxplot(column=X_col[i], by='y', ax=ax)
   plt.show()