%reset -f
#%% import
import os
os.getcwd()
from os import chdir
os.chdir('C:\\Users\\UOS\\Documents\\GITHUB\LFU')
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

#%% expand output display pd.df
pd.set_option('display.expand_frame_repr', False) 

#%% Load raw data
df_buy  = pd.read_csv("C:\\Users\\UOS\\Dropbox\\LFY\\datasets\\df_buy.csv",index_col=0) 
df_nobuy = pd.read_csv("C:\\Users\\UOS\\Dropbox\\LFY\\datasets\\df_no_buy.csv",index_col=0) 

#%% 접속 시간 차이
hit_pss_tm_buy = df_buy.drop_duplicates(['clnt_id','sess_id','sess_dt','tot_sess_hr_v'])['tot_sess_hr_v']
hit_pss_tm_nobuy = df_nobuy.drop_duplicates(['clnt_id','sess_id','sess_dt','tot_sess_hr_v'])['tot_sess_hr_v']

figure, (ax1, ax2) = plt.subplots(1,2)
figure.set_size_inches(10,5)
ax1.hist(hit_pss_tm_buy) ; ax2.hist(hit_pss_tm_nobuy) 

figure, (ax1, ax2) = plt.subplots(1,2)
figure.set_size_inches(10,5)
ax1.hist(hit_pss_tm_buy[hit_pss_tm_buy<100]) ; ax2.hist(hit_pss_tm_nobuy[hit_pss_tm_nobuy<100]) 

pd.DataFrame(hit_pss_tm_buy).boxplot(showfliers=False)
pd.DataFrame(hit_pss_tm_nobuy).boxplot(showfliers=False)

hit_pss_tm_buy.mean()
hit_pss_tm_buy.median()
hit_pss_tm_nobuy.mean()
hit_pss_tm_nobuy.median()


#%% 총페이지 조회수
tot_view_buy = df_buy.drop_duplicates(['clnt_id','sess_id','sess_dt','tot_pag_view_ct'])['tot_pag_view_ct']
tot_view_nobuy = df_nobuy.drop_duplicates(['clnt_id','sess_id','sess_dt','tot_pag_view_ct'])['tot_pag_view_ct']

figure, (ax1, ax2) = plt.subplots(1,2)
figure.set_size_inches(10,5)
ax1.hist(tot_view_buy) ; ax2.hist(tot_view_nobuy) 

figure, (ax1, ax2) = plt.subplots(1,2)
figure.set_size_inches(10,5)
ax1.hist(tot_view_buy[tot_view_buy<100]) ; ax2.hist(tot_view_nobuy[tot_view_nobuy<100]) 

pd.DataFrame(tot_view_buy).boxplot(showfliers=False)
pd.DataFrame(tot_view_nobuy).boxplot(showfliers=False)

tot_view_buy.mean()
tot_view_buy.median()
tot_view_nobuy.mean()
tot_view_nobuy.median()

#%% 접속시간과 관련있는 변수는?
df_buy_tmp = df_buy.drop_duplicates(['clnt_id','sess_id','sess_dt','tot_sess_hr_v'])         # 범주분석을 위해 하나의 행씩만 추출해놓자
df_nobuy_tmp = df_nobuy.drop_duplicates(['clnt_id','sess_id','sess_dt','tot_sess_hr_v'])

sns.catplot(x="biz_unit", y="tot_sess_hr_v", kind="box", data=df_buy_tmp, showfliers=False)
sns.catplot(x="biz_unit", y="tot_sess_hr_v", kind="box", data=df_nobuy_tmp, showfliers=False)

sns.catplot(x="trfc_src", y="tot_sess_hr_v", kind="box", data=df_buy_tmp, showfliers=False)
sns.catplot(x="trfc_src", y="tot_sess_hr_v", kind="box", data=df_nobuy_tmp, showfliers=False)

sns.catplot(x="dvc_ctg_nm", y="tot_sess_hr_v", kind="box", data=df_buy_tmp, showfliers=False)
sns.catplot(x="dvc_ctg_nm", y="tot_sess_hr_v", kind="box", data=df_nobuy_tmp, showfliers=False)

sns.catplot(x="clnt_gender", y="tot_sess_hr_v", kind="box", data=df_buy_tmp, showfliers=False)
sns.catplot(x="clnt_gender", y="tot_sess_hr_v", kind="box", data=df_nobuy_tmp, showfliers=False)

sns.catplot(x="clnt_age", y="tot_sess_hr_v", kind="box", data=df_buy_tmp, showfliers=False)
sns.catplot(x="clnt_age", y="tot_sess_hr_v", kind="box", data=df_nobuy_tmp, showfliers=False)


#%% 범주데이터
buy_by_biz = df_buy_tmp.groupby('biz_unit')['id'].agg('count')
buy_by_trfc = df_buy_tmp.groupby('trfc_src')['id'].agg('count')
buy_by_dvc = df_buy_tmp.groupby('dvc_ctg_nm')['id'].agg('count')
buy_by_clnt_gender = df_buy_tmp.groupby('clnt_gender')['id'].agg('count')
label = buy_by_biz.index; index = np.arange(len(label)) ;plt.bar(index, buy_by_biz) ; plt.xticks(index, label, fontsize=15)
label = buy_by_trfc.index; index = np.arange(len(label)) ;plt.bar(index,  buy_by_trfc) ; plt.xticks(index, label, fontsize=15,rotation=45)
label = buy_by_dvc.index; index = np.arange(len(label)) ;plt.bar(index,  buy_by_dvc) ; plt.xticks(index, label, fontsize=15)
label = buy_by_clnt_gender.index; index = np.arange(len(label)) ;plt.bar(index,  buy_by_clnt_gender) ; plt.xticks(index, label, fontsize=15)

nobuy_by_biz = df_nobuy_tmp.groupby('biz_unit')['id'].agg('count')
nobuy_by_trfc= df_nobuy_tmp.groupby('trfc_src')['id'].agg('count')
nobuy_by_dvc = df_nobuy_tmp.groupby('dvc_ctg_nm')['id'].agg('count')
nobuy_by_clnt_gender = df_nobuy_tmp.groupby('clnt_gender')['id'].agg('count')
label = nobuy_by_biz.index; index = np.arange(len(label)) ;plt.bar(index, nobuy_by_biz) ; plt.xticks(index, label, fontsize=15)
label = nobuy_by_trfc.index; index = np.arange(len(label)) ;plt.bar(index, nobuy_by_trfc) ; plt.xticks(index, label, fontsize=15,rotation=45)
label = nobuy_by_dvc.index; index = np.arange(len(label)) ;plt.bar(index, nobuy_by_dvc) ; plt.xticks(index, label, fontsize=15)
label = nobuy_by_clnt_gender.index; index = np.arange(len(label)) ;plt.bar(index, nobuy_by_clnt_gender) ; plt.xticks(index, label, fontsize=15)

# biz_unit -> trfc_src
tmp=df_buy_tmp[df_buy_tmp['biz_unit']=='A01'].groupby('trfc_src')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_buy_tmp[df_buy_tmp['biz_unit']=='A02'].groupby('trfc_src')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_buy_tmp[df_buy_tmp['biz_unit']=='A03'].groupby('trfc_src')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)

tmp=df_nobuy_tmp[df_nobuy_tmp['biz_unit']=='A01'].groupby('trfc_src')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['biz_unit']=='A02'].groupby('trfc_src')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['biz_unit']=='A03'].groupby('trfc_src')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)

# biz_unit -> dvc_ctg_nm
tmp=df_buy_tmp[df_buy_tmp['biz_unit']=='A01'].groupby('dvc_ctg_nm')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_buy_tmp[df_buy_tmp['biz_unit']=='A02'].groupby('dvc_ctg_nm')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_buy_tmp[df_buy_tmp['biz_unit']=='A03'].groupby('dvc_ctg_nm')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)

tmp=df_nobuy_tmp[df_nobuy_tmp['biz_unit']=='A01'].groupby('dvc_ctg_nm')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['biz_unit']=='A02'].groupby('dvc_ctg_nm')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['biz_unit']=='A03'].groupby('dvc_ctg_nm')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)

# biz_unit -> clnt_gender
tmp=df_buy_tmp[df_buy_tmp['biz_unit']=='A01'].groupby('clnt_gender')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_buy_tmp[df_buy_tmp['biz_unit']=='A02'].groupby('clnt_gender')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_buy_tmp[df_buy_tmp['biz_unit']=='A03'].groupby('clnt_gender')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)

tmp=df_nobuy_tmp[df_nobuy_tmp['biz_unit']=='A01'].groupby('clnt_gender')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['biz_unit']=='A02'].groupby('clnt_gender')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['biz_unit']=='A03'].groupby('clnt_gender')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)

# biz_unit -> clnt_age
tmp=df_buy_tmp[df_buy_tmp['biz_unit']=='A01'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_buy_tmp[df_buy_tmp['biz_unit']=='A02'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_buy_tmp[df_buy_tmp['biz_unit']=='A03'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)

tmp=df_nobuy_tmp[df_nobuy_tmp['biz_unit']=='A01'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['biz_unit']=='A02'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['biz_unit']=='A03'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)

# biz_unit -> tot_pag_view_ct
sns.catplot(x="biz_unit", y="tot_pag_view_ct", kind="box", data=df_buy_tmp, showfliers=False)
sns.catplot(x="biz_unit", y="tot_pag_view_ct", kind="box", data=df_nobuy_tmp, showfliers=False)


# trfc_src -> dvc_ctg_nm
tmp=df_buy_tmp[df_buy_tmp['trfc_src']=='DIRECT'].groupby('dvc_ctg_nm')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_buy_tmp[df_buy_tmp['trfc_src']=='PUSH'].groupby('dvc_ctg_nm')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_buy_tmp[df_buy_tmp['trfc_src']=='unknown'].groupby('dvc_ctg_nm')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_buy_tmp[df_buy_tmp['trfc_src']=='PORTAL_1'].groupby('dvc_ctg_nm')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_buy_tmp[df_buy_tmp['trfc_src']=='PORTAL_2'].groupby('dvc_ctg_nm')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_buy_tmp[df_buy_tmp['trfc_src']=='PORTAL_3'].groupby('dvc_ctg_nm')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_buy_tmp[df_buy_tmp['trfc_src']=='WEBSITE'].groupby('dvc_ctg_nm')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)

tmp=df_nobuy_tmp[df_nobuy_tmp['trfc_src']=='DIRECT'].groupby('dvc_ctg_nm')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['trfc_src']=='PUSH'].groupby('dvc_ctg_nm')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['trfc_src']=='unknown'].groupby('dvc_ctg_nm')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['trfc_src']=='PORTAL_1'].groupby('dvc_ctg_nm')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['trfc_src']=='PORTAL_2'].groupby('dvc_ctg_nm')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['trfc_src']=='PORTAL_3'].groupby('dvc_ctg_nm')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['trfc_src']=='WEBSITE'].groupby('dvc_ctg_nm')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)


# trfc_src -> clnt_age
tmp=df_buy_tmp[df_buy_tmp['trfc_src']=='DIRECT'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_buy_tmp[df_buy_tmp['trfc_src']=='PUSH'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_buy_tmp[df_buy_tmp['trfc_src']=='WEBSITE'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_buy_tmp[df_buy_tmp['trfc_src']=='unknown'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_buy_tmp[df_buy_tmp['trfc_src']=='PORTAL_1'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_buy_tmp[df_buy_tmp['trfc_src']=='PORTAL_2'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_buy_tmp[df_buy_tmp['trfc_src']=='PORTAL_3'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)

tmp=df_nobuy_tmp[df_nobuy_tmp['trfc_src']=='DIRECT'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['trfc_src']=='PUSH'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['trfc_src']=='WEBSITE'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['trfc_src']=='unknown'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['trfc_src']=='PORTAL_1'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['trfc_src']=='PORTAL_2'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['trfc_src']=='PORTAL_3'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)


# trfc_src -> clnt_gender
tmp=df_nobuy_tmp[df_nobuy_tmp['trfc_src']=='DIRECT'].groupby('clnt_gender')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['trfc_src']=='PUSH'].groupby('clnt_gender')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['trfc_src']=='WEBSITE'].groupby('clnt_gender')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['trfc_src']=='unknown'].groupby('clnt_gender')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['trfc_src']=='PORTAL_1'].groupby('clnt_gender')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['trfc_src']=='PORTAL_2'].groupby('clnt_gender')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['trfc_src']=='PORTAL_3'].groupby('clnt_gender')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)


# trfc_src -> tot_pag_view_ct
sns.catplot(x="trfc_src", y="tot_pag_view_ct", kind="box", data=df_buy_tmp, showfliers=False)
sns.catplot(x="trfc_src", y="tot_pag_view_ct", kind="box", data=df_nobuy_tmp, showfliers=False)

# dvc_ctg_nm -> clnt_age
tmp=df_buy_tmp[df_buy_tmp['dvc_ctg_nm']=='PC'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_buy_tmp[df_buy_tmp['dvc_ctg_nm']=='mobile_app'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_buy_tmp[df_buy_tmp['dvc_ctg_nm']=='mobile_web'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)

tmp=df_nobuy_tmp[df_nobuy_tmp['dvc_ctg_nm']=='PC'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['dvc_ctg_nm']=='mobile_app'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)
tmp=df_nobuy_tmp[df_nobuy_tmp['dvc_ctg_nm']=='mobile_web'].groupby('clnt_age')['id'].agg('count')
label = tmp.index; index = np.arange(len(label)) ;plt.bar(index,tmp) ; plt.xticks(index, label, fontsize=15)


#%% action_type 관계
# 구매고객
total_hit_seq = df_buy.sort_values(['clnt_id','sess_id','sess_dt','hit_seq']).groupby(['clnt_id','sess_id','sess_dt']).nth(-1)['hit_seq'] # 마지막 행동
count_hit_seq_tmp = df_buy.groupby(['clnt_id','sess_id','sess_dt','action_type'])['hit_seq'].agg('count')
freq_hit_seq_tmp = count_hit_seq_tmp/total_hit_seq

freq_hit_seq = freq_hit_seq_tmp.reset_index().groupby(['action_type']).agg('mean')
label = freq_hit_seq.index; index = np.arange(len(label)) ;plt.bar(index, freq_hit_seq.hit_seq) ; plt.xticks(index, label, fontsize=15)
count_hit_seq = count_hit_seq_tmp.reset_index().groupby(['action_type']).agg('mean')
label = count_hit_seq.index; index = np.arange(len(label)) ;plt.bar(index, count_hit_seq.hit_seq) ; plt.xticks(index, label, fontsize=15)

# 비구매고객
total_hit_seq = df_nobuy.sort_values(['clnt_id','sess_id','sess_dt','hit_seq']).groupby(['clnt_id','sess_id','sess_dt']).nth(-1)['hit_seq'] # 마지막 행동
count_hit_seq_tmp = df_nobuy.groupby(['clnt_id','sess_id','sess_dt','action_type'])['hit_seq'].agg('count')
freq_hit_seq_tmp = count_hit_seq_tmp/total_hit_seq

freq_hit_seq = freq_hit_seq_tmp.reset_index().groupby(['action_type']).agg('mean')
label = freq_hit_seq.index; index = np.arange(len(label)) ;plt.bar(index, freq_hit_seq.hit_seq) ; plt.xticks(index, label, fontsize=15)
count_hit_seq = count_hit_seq_tmp.reset_index().groupby(['action_type']).agg('mean')
label = count_hit_seq.index; index = np.arange(len(label)) ;plt.bar(index, count_hit_seq.hit_seq) ; plt.xticks(index, label, fontsize=15)




#%% 1222 구매 전 행동들에 시간을 얼마나 사용하는가?
df_buy['hit_diff']=df_buy.sort_values(['clnt_id','sess_id','sess_dt','hit_seq']).groupby(['clnt_id','sess_id','sess_dt'])['hit_pss_tm'].diff(periods=-1)*-1
#df_buy.sort_values(['clnt_id','sess_id','sess_dt','hit_seq']).groupby(['clnt_id','sess_id','sess_dt']).head()
hit_diff_tmp1 = df_buy.groupby(['clnt_id','sess_id','sess_dt','action_type'])['hit_diff'].agg('sum') # 세션마다 각 행동에 드는 시간의 합
hit_diff = hit_diff_tmp1.reset_index().groupby(['action_type']).agg('mean')
label = hit_diff.index; index = np.arange(len(label)) ;plt.bar(index, hit_diff.hit_diff) ; plt.xticks(index, label, fontsize=15)

df_buy['hit_diff_ratio'] = df_buy['hit_diff']/ df_buy['tot_sess_hr_v'] 
hit_diff_ratio = df_buy.groupby(['action_type'])['hit_diff_ratio'].agg('mean')
label = hit_diff_ratio.index; index = np.arange(len(label)) ;plt.bar(index, hit_diff_ratio) ; plt.xticks(index, label, fontsize=15)







# 들어오는 경로마다 첫번째 action이 뭔지?


# 검색키워드가 있는지 차이
d1_tmp5 = df_buy.sort_values(['clnt_id','sess_id','sess_dt','action_type']).groupby(['clnt_id','sess_id','sess_dt'],as_index=False).nth(0)
pd.isnull(d1_tmp5['sech_kwd']).sum() / d1_tmp5.shape[0] # 구매를 한 사람중에 키워드가 있는 비율
#notnull
d1_tmp6 = df_nobuy.sort_values(['clnt_id','sess_id','sess_dt','action_type']).groupby(['clnt_id','sess_id','sess_dt'],as_index=False).nth(0)
pd.isnull(d1_tmp6['sech_kwd']).sum() / d1_tmp6.shape[0] # 구매를 안 한 사람중에 키워드가 있는 비율



#%% 총 행동 수 
hit_seq_buy = df_buy.drop_duplicates(['clnt_id','sess_id','sess_dt','tot_sess_hr_v'])['hit_seq']
hit_seq_nobuy = df_nobuy.drop_duplicates(['clnt_id','sess_id','sess_dt','tot_sess_hr_v'])['hit_seq']


#%% 1222 같은 session 내에서 검색을 한 상품을 구매할 확률은?
df_buy_sort = df_buy.sort_values(['clnt_id','sess_id','sess_dt','action_type'])
df_buy_grouped = df_buy_sort.groupby(['clnt_id','sess_id','sess_dt'],as_index=False).nth(0)
sech_T_key = df_buy_grouped[df_buy_grouped['sech_kwd'].notnull()]  # sech_kwd 존재하고 구매를 함

data2.rename(columns={'de_dt':'sess_dt'}, inplace=True)
data_buy_tmp = pd.merge(df_buy, data2, on=['clnt_id','sess_dt','biz_unit']) #  거래데이터 & data2 에 동시에 존재하는 데이터만 고려

#%% 1222 A01, A02, A03에 대해 알아보자
#data2.rename(columns={'de_dt':'sess_dt'}, inplace=True)
#data_buy_tmp = pd.merge(sech_T_key, data2, on=['clnt_id','sess_dt','biz_unit']) #  거래데이터 & data2 에 동시에 존재하는 데이터만 고려
# A03
#tmp_key = data_buy_tmp[data_buy_tmp['biz_unit']=="A03"].iloc[-1]; tmp_key
#trans_key = tmp_key.trans_id_y
#pd_key = data_buy_tmp [data_buy_tmp.trans_id_y == trans_key]
#pd_c_key = pd_key.pd_c.astype('int64').to_frame()
#data4[data4['pd_c'].isin(pd_c_key.pd_c)]

#%% miss


d1_tmp5 = data1_trans_F[['clnt_id','sess_id','sess_dt']].drop_duplicates() 
pd.merge(data1_trans_F, d1_tmp5, how="right", on=['clnt_id','sess_id','sess_dt'])

d1_tmp6 = data1_trans_F[data1_trans_F['action_type']==0]
d1_tmp6.groupby(['clnt_id','sess_id','sess_dt']).nth(0) 



# help : ctrl+I or 함수()
# 검색키워드가있는 / 그룹별로 / 구매까지 걸린 시간 봐도 ㄱㅊ을듯

df = pd.DataFrame({"A":["foo", "foo", "foo", "bar"], "B":[0,1,1,1], "C":["A","A","B","A"]})
df
df.drop_duplicates(subset=['A', 'C'], keep=False)

# clnt_id, see_id, sess_dt 가 일치하는 자료만 뽑는다



df_buy.info()

# session 접속 시간이 긴데 구매를 하는 고객과 구매를 하지 않는 고객의 차이 



df_buy["clnt_id"]
# 거래를 한 사람중에 kwd가 존재하는 사람 (같은 sess_id 내에서 찾아야함)

