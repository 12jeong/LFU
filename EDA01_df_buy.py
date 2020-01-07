%reset -f
# import
import os
os.getcwd()
from os import chdir
#os.chdir('C:\\Users\\UOS\\Documents\\GITHUB\LFU')
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

#
pd.set_option('display.expand_frame_repr', False) # expand output display pd.df

# Load raw data
raw_online_bh  = pd.read_csv(".\Dropbox\\LFY\\datasets\\rawdata\\online_bh.csv")   #  3196362
raw_trans_info = pd.read_csv(".\Dropbox\\LFY\\datasets\\rawdata\\trans_info.csv") 
raw_demo_info  = pd.read_csv(".\Dropbox\\LFY\\datasets\\rawdata\\demo_info.csv") 
raw_prod_info  = pd.read_csv(".\Dropbox\\LFY\\datasets\\rawdata\\prod_info.csv") 

# copy raw data
online_bh = raw_online_bh.copy()
trans_info= raw_trans_info.copy()
demo_info = raw_demo_info.copy()
prod_info = raw_prod_info.copy()

# data1 & data3 merge
df_right = [raw_demo_info]
for dtidx in range(len(df_right)):
    dt_temp = df_right[dtidx].copy()
    online_bh = online_bh.merge(dt_temp, on='clnt_id', how='left')

# 2,3,4 merge
prod_info_uk=raw_prod_info.copy()
prod_info_uk['pd_c'] = prod_info_uk['pd_c'].astype(str).str.zfill(4)
prod_info_uk.loc[1668]=['unknown','unknown','unknown','unknown']

trans_info = raw_trans_info.copy()
df_right = [raw_demo_info,prod_info_uk]
key = ['clnt_id','pd_c']
how = ['left','left']
for dtidx in range(len(df_right)):
    dt_temp = df_right[dtidx].copy().drop_duplicates()
    trans_info = trans_info.merge(dt_temp, on=key[dtidx], how=how[dtidx])

trans_info.head()

# data1 : online_bh 전처리 
online_bh.head()
online_bh["id"] = online_bh.index
online_bh["hit_pss_tm"] = online_bh["hit_pss_tm"]/(60*1000)   # 분으로 변환
online_bh["tot_sess_hr_v"] = online_bh["tot_sess_hr_v"]/60    # 분으로 변환

# action_type == 6 or 7 인데 trans_id == NaN 인 행 제거
sum(np.isnan(online_bh[(online_bh.action_type == 6) | (online_bh.action_type == 7) ].trans_id)) 
online_bh = online_bh[~( (np.isnan(online_bh.trans_id) == True) & ((online_bh.action_type==6)| (online_bh.action_type == 7)))]
sum(np.isnan(online_bh[(online_bh.action_type == 6) | (online_bh.action_type == 7) ].trans_id)) 

# action_type == 7 & trans_id 중복된 행, 첫 행만 보존
df_action7 =online_bh.copy()
df_action7 = df_action7[df_action7['action_type']==7]
df_action7.sort_values(['clnt_id','sess_id','sess_dt','hit_seq']).groupby(['clnt_id','sess_id','sess_dt']).head()
df_action7_nd = df_action7.drop_duplicates(['trans_id'],keep="first")
df_action7_nd.sort_values(['clnt_id','sess_id','sess_dt','hit_seq']).groupby(['clnt_id','sess_id','sess_dt']).head()
df_action_n7 =online_bh.copy()
df_action_n7 = df_action_n7[df_action_n7['action_type']!=7]
online_bh=pd.concat([df_action7_nd, df_action_n7]) 

# astype == str
online_bh['clnt_id']=online_bh['clnt_id'].astype('str')
online_bh['sess_id']=online_bh['sess_id'].astype('str')
online_bh['trans_id'] =online_bh['trans_id'].astype('str')
online_bh['sess_dt'] = online_bh[['sess_dt']].applymap(str).applymap(lambda s: "{}-{}-{}".format(s[0:4],s[4:6],s[6:],))
online_bh['sess_dt'] = pd.to_datetime(online_bh['sess_dt'])

# data2 : trans_info 전처리

# -- 구매 수량 (buy_ct) == 0 인거 존재
trans_info['buy_ct'].unique()
trans_info[trans_info['buy_ct'] ==0]['biz_unit'].unique() # A03 뿐
trans_info[trans_info['buy_ct']==500]['biz_unit'].unique() # A03 뿐

trans_info = trans_info[~trans_info.buy_ct.isin([0,500])]

# pd_c = NaN 제거
trans_info = trans_info[~(trans_info['pd_c'] == 'unknown')] 

# -- 구매 금액 천억 이상 : Toilet Papers , unknown (bill100_lines보면 이사람이 애초에 비쌈걸 사는 사람도 아니고 A03임) : 제거
trans_info['buy_am'].describe()
sort_am = trans_info.sort_values(['buy_am'],ascending=False)
bill100_key = trans_info[trans_info.buy_am.isin([100000016899,100000007199])][['clnt_id','trans_id','de_dt','de_tm']] 
bill100_lines = pd.merge(trans_info,bill100_key, how='right',on=['clnt_id','trans_id','de_dt','de_tm']).drop_duplicates()
trans_info = trans_info[~trans_info.buy_am.isin([100000016899,100000007199])]

# 1.1 구매일자 : 'de_dt' - 일자별 ->  요일별 추이
trans_info['de_dt'] = trans_info[['de_dt']].applymap(str).applymap(lambda s: "{}-{}-{}".format(s[0:4],s[4:6],s[6:],))
trans_info['de_dt'] = pd.to_datetime(trans_info['de_dt'])
trans_info['day_of_week'] = trans_info['de_dt'].dt.day_name()
trans_info['weekend'] = 0
trans_info['weekend'][trans_info.day_of_week.isin(['Saturday','Sunday'])] = 1

# astype == str
trans_info['clnt_id']=trans_info['clnt_id'].astype('str')
trans_info['trans_id']=trans_info['trans_id'].astype('int').astype('str')


#nobuy 정의 : 동일 session내에서 action_type = 6(구매 완료) 없는 사람 
df_buy =online_bh.copy()
df_buy = df_buy[df_buy['action_type']==6]

buy_session_key = df_buy[['clnt_id','sess_id','sess_dt']]
buy_session_key['buy']=1
buy_session_key

temp = online_bh.copy()
temp=temp.merge(buy_session_key, how='left').drop_duplicates() # 중복행 제거해야함

temp.head()

df_nobuy =temp[temp['buy'].isna()]
df_buy = temp[temp['buy'].notnull()]

df_nobuy.shape[0] + df_buy.shape[0] - online_bh.shape[0] # 행 개수 확인

# buy 정의 : 같은 session 내에서 구매후 환불을 하지 않은 사람 (노이즈 제거) - 추후 환불 고객 분석 가능

buy_tmp = df_buy.copy()
buy_tmp2 = buy_tmp.groupby(['clnt_id','sess_id','sess_dt','trans_id'])['id'].agg(trans_count='count') < 2
no_cancel_key = buy_tmp2[ buy_tmp2['trans_count'].notna()].reset_index()[['clnt_id','sess_id','sess_dt']]
buy_tmp   = buy_tmp.merge(no_cancel_key, how='inner').drop_duplicates() 
sum(buy_tmp.groupby(['clnt_id','sess_id','sess_dt','trans_id'])['id'].agg('count') > 1) # 이 경우는 같은세션에서 다른 상품을 구매한 것
df_buy = buy_tmp

# 다른 session 인데 같은 tm 에 trans_id가 중복되는 경우 hit_pss_tm 이 큰 행만 살린다.
df_tmp = buy_tmp[buy_tmp['action_type']==6]
ttmp = (df_tmp.groupby(['clnt_id','sess_dt','hit_tm','trans_id'])['id'].agg({'tmp_overlap':'count'})>1).reset_index()
tkey = ttmp[ttmp.tmp_overlap==1].iloc[:,0:4]
tmerge = pd.merge(df_buy, tkey, how='inner')
tmerge2 = tmerge.sort_values(['clnt_id','sess_dt','sess_id','hit_pss_tm']).groupby(['clnt_id','sess_dt','hit_tm','trans_id'],as_index=False).nth(-1)

df_buy = df_buy[~(df_buy['id'].isin(tmerge2['id']))]

# 저장
online_bh = online_bh.sort_values(['clnt_id','sess_dt','sess_id','hit_seq'])
trans_info = trans_info.sort_values(['clnt_id','de_dt','de_tm','trans_seq'])

online_bh.to_csv(pdir+"\\Dropbox\\LFY\\datasets\\online_bh.csv",index=False)
trans_info.to_csv(pdir+"\\Dropbox\\LFY\\datasets\\trans_info.csv",index=False)
df_buy.csv(pdir+"\\Dropbox\\LFY\\datasets\\df_buy.csv",index=False)
df_nobuy.csv(pdir+"\\Dropbox\\LFY\\datasets\\df_nobuy.csv",index=False)