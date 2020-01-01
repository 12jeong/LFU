#%%  EDA plot for "trans_bh"
# "trans_bh" 데이터 (설명) : df_buy['action_type'] == 6 인 사람들의 id,time 정보로 trans_info 데이터 subset

# --- 층화
# 01 ------------ 
# 구매일자 : 'de_dt' - 일자별 ->  요일별 추이
# 구매 시간 : 'de_tm', - 시간대별 추이(산점도) -> 군집화로 나눠서 보기 
### new data
# *요일 : 'day_of_week'
# *주말 : 'weekend' 
# 공휴일 :

# 02 ------------  
# *업종 단위 : biz_unit',

# 03 ------------ ??
# 상품 코드 : 'pd_c', clac_nm1','clac_nm2', 'clac_nm3'
# - muti '업종 별 상품 코드!! : biz_unit',

# 04 ------------
# *인적 정보 : 'clnt_gender', 'clnt_age', 
# - muti 성별별 나이 : 'clnt_gender', 'clnt_age'

# ---- Y
#  01.  단순 빈도
#  02. 구매 순서  : trans_seq', -> 내역 내 구매 순서 -> 경로적인 의미 있을 수 있음 
#  03. 구매 금액, 수량 : buy_am', 'buy_ct', 

#%%
%reset -f
#%% import
import os
os.getcwd()
from os import chdir
pc12 = "C:\\Users\\UOS\\"
pcsh = "C:\\Users\\user\\"
# pcyt = ""
pc = pcsh

os.chdir(pc+'Documents\\GITHUB\LFU')

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
import seaborn as sns
mpl.rcParams.update({'font.size':14})
plt.rc('font',family='Malgun Gothic') # windows
%matplotlib inline 

#%% expand output display pd.df
pd.set_option('display.expand_frame_repr', False) 

#%% Load raw data
raw_df_refund    = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\df_refund.csv") 
raw_df_norefund  = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\df_norefund.csv") 
raw_df_buy       = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\df_buy.csv",index_col=0) 
raw_df_nobuy     = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\df_no_buy.csv",index_col=0) 
raw_trans_info   = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\mg_trans_info.csv")
raw_online_bh    = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\mg_online_bh.csv")


df_refund   = raw_df_refund.copy()  
df_norefund = raw_df_norefund.copy()  
df_buy      = raw_df_buy.copy()  
df_nobuy    = raw_df_nobuy.copy()
trans_info  = raw_trans_info.copy()
online_bh   = raw_online_bh.copy()

#%% trans_bh - df_buy['action_type'] == 6 인 사람들의 id,time 정보로 trans_info 데이터 subset
buy_action_key = df_buy[df_buy['action_type'] == 6][['clnt_id','trans_id','sess_dt','hit_tm']]
buy_action_key['obh'] = 1

temp = trans_info.copy()
trans_info = pd.merge(temp,buy_action_key, how='left',
                         left_on=['clnt_id','trans_id','de_dt','de_tm'],
                         right_on=['clnt_id','trans_id','sess_dt','hit_tm']).drop_duplicates()

col_order = ['obh', 'clnt_id', 'trans_id', 'sess_dt','de_dt', 'hit_tm','de_tm',
             'trans_seq', 'biz_unit', 'pd_c', 'buy_am', 'buy_ct', 'clnt_gender', 'clnt_age', 'clac_nm1','clac_nm2', 'clac_nm3']
trans_info=trans_info[col_order]

trans_bh = trans_info[trans_info['obh']==1]
trans_only = trans_info[trans_info['obh'].isna()]
trans_bh['sess_dt'] = trans_bh['sess_dt'].astype(int)

trans_bh = trans_bh.sort_values(['obh', 'clnt_id', 'trans_id', 'sess_dt', 'de_dt', 'hit_tm', 'de_tm','trans_seq'],axis=0) # 오름차순
#sum(trans_bh['sess_dt'] != trans_bh['de_dt'])
#sum(trans_bh['hit_tm'] != trans_bh['de_tm'])
trans_bh = trans_bh.drop(['sess_dt','hit_tm'],axis=1)
trans_bh['freq']=1
trans_bh.to_csv(pc+"Dropbox\\LFY\\datasets\\trans_bh.csv", index=False)

#%%  EDA plot for "trans_bh"
raw_trans_bh = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\trans_bh.csv")

# -- 구매 수량 == 0 인거 존재
raw_trans_bh['buy_ct'].unique()
raw_trans_bh[raw_trans_bh['buy_ct'] ==0]['biz_unit'].unique() # A03 뿐
raw_trans_bh[raw_trans_bh['buy_ct']==500]['biz_unit'].unique() # A03 뿐

trans_bh = raw_trans_bh[~raw_trans_bh.buy_ct.isin([0,500])]

# -- 구매 금액 천억 이상 : Toilet Papers , unknown (bill100_lines보면 이사람이 애초에 비쌈걸 사는 사람도 아니고 A03임) : 제거
trans_bh['buy_am'].describe()
sort_am = raw_trans_bh.sort_values(['buy_am'],ascending=False)
bill100_key = trans_bh[trans_bh.buy_am.isin([100000016899,100000007199])][['clnt_id','trans_id','de_dt','de_tm']] 
bill100_lines = pd.merge(trans_bh,bill100_key, how='right',on=['clnt_id','trans_id','de_dt','de_tm']).drop_duplicates()
trans_bh = trans_bh[~trans_bh.buy_am.isin([100000016899,100000007199])]

# 1.1 구매일자 : 'de_dt' - 일자별 ->  요일별 추이
trans_bh['de_dt_str'] = trans_bh[['de_dt']].applymap(str).applymap(lambda s: "{}-{}-{}".format(s[0:4],s[4:6],s[6:],))
trans_bh['de_dt_str'] = pd.to_datetime(trans_bh['de_dt_str'])
trans_bh['day_of_week'] = trans_bh['de_dt_str'].dt.day_name()
trans_bh['weekend'] = 0
trans_bh['weekend'][trans_bh.day_of_week.isin(['Saturday','Sunday'])] = 1


# -- 공휴일 정보 - API https://blog.naver.com/hancury/221057426711 
# from urllib2 import Request, urlopen
# from urllib import urlencode, quote_plus

# url = 'http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getSundryDayInfo'
# queryParams = '?' + urlencode({ quote_plus('ServiceKey') : '서비스키', quote_plus('solYear') : '2015', quote_plus('solMonth') : '10' })

# request = Request(url + queryParams)
# request.get_method = lambda: 'GET'
# response_body = urlopen(request).read()
# print response_body
trans_bh.to_csv(pc+"Dropbox\\LFY\\datasets\\pp_trans_bh.csv", index=False)

#%%
#%% function for plot
# 01 단순 빈도
def plot_freq(dataset,var,ytype,xsize,ysize) :
    if ytype == "freq":
        y = frequency = dataset.groupby(var).freq.sum()
    elif ytype == "prop":
        y = proportion = dataset.groupby(var).freq.sum() / dataset.freq.sum() *100
        
    label = y.index
    plt.style.use('ggplot')

    # -- font
    fig = plt.figure(figsize=(xsize,ysize))
    ax = fig.add_subplot(111)
    #plt.rcParams['figure.figsize'] = [xsize,ysize]
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica'
    # -- style of axes
    plt.rcParams['axes.edgecolor'] = '#333F4B'
    plt.rcParams['axes.linewidth'] = 0.8
    
    # == main part
    plot = plt.bar(label, y.values, color='#007acc', alpha=0.8)    
    plt.title("[trans_bh] Frequncies of "+ var ,fontsize= 16)
    plt.xlabel('Type of '+ var, fontsize=15)
    if ytype == "freq":
        plt.ylabel('Frequency of '+ var, fontsize=15)
    elif ytype == "prop":
        plt.ylabel('Proportion of '+ var, fontsize=15)
    
    for i, rect in enumerate(plot):       
        ax.text(rect.get_x() + rect.get_width() / 1.3, 0.95 * rect.get_height(), str(round(y[i],2)) + '%', ha='right', va='center')
        

        
#%% 전처리된 trans_info 불러오기
trans_bh = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\pp_trans_bh.csv")
#%%
# 02 ------------  
# 2. '업종 단위 : biz_unit',
# ---- Y
#  01.  단순 빈도
plot_freq(trans_bh,'biz_unit','prop',5,5)

A01_trans = trans_bh[trans_bh['biz_unit']=='A01']
A02_trans = trans_bh[trans_bh['biz_unit']=='A02']
A03_trans = trans_bh[trans_bh['biz_unit']=='A03']

#  02. 구매 순서  : trans_seq', -> 내역 내 구매 순서 -> 경로적인 의미 있을 수 있음 


#  03. 구매 금액, 수량 : buy_am', 'buy_ct', 
# -- 구매 금액
# ------------ 층화 : biz_unit, 나이, 성별
sns.catplot(x="clnt_gender", y="buy_am", hue="clnt_age", col="biz_unit",data=trans_bh, kind="bar",height=4, aspect=.7)
sns.catplot(x="clnt_age", y="buy_am", hue="clnt_gender", col="biz_unit",data=trans_bh, kind="bar",height=4, aspect=.7)

sns.FacetGrid(trans_bh, col="biz_unit", height=4, aspect=.5).map(sns.barplot, "clnt_age", "buy_am");
sns.FacetGrid(trans_bh, col="biz_unit", height=4, aspect=.5).map(sns.barplot, "clnt_gender", "buy_am");


sns.catplot(x="biz_unit", y="buy_am", kind="box", data=trans_bh, showfliers=False)
sns.boxplot(x="biz_unit", y="buy_am", hue="clnt_gender", data=trans_bh, palette="Set1", showfliers=False)
sns.boxplot(x="biz_unit", y="buy_am", hue="clnt_age", data=trans_bh, palette="Set1", showfliers=False)

#sns.FacetGrid(trans_bh, col="biz_unit", height=4, aspect=.5).map(sns.boxplot, "clnt_gender", "buy_am", showfliers=False);

sns.catplot(x="clnt_gender", y="buy_am", kind="box", data=trans_bh, showfliers=False)
sns.boxplot(x="clnt_gender", y="buy_am", hue="biz_unit", data=trans_bh, palette="Set1", showfliers=False)

sns.catplot(x="clnt_age", y="buy_am", kind="box", data=trans_bh, showfliers=False)
sns.boxplot(x="clnt_age", y="buy_am", hue="biz_unit", data=trans_bh, palette="Set1", showfliers=False)


# -- 구매 수량
plot_freq(trans_bh,'buy_ct','prop',10,5)

trans_bh['buy_ct'].unique()
np.around(trans_bh.groupby('buy_ct').freq.sum() / trans_bh.freq.sum() *100,4).sort_values(ascending=False)


sns.barplot(x='biz_unit', y='buy_ct', hue='clnt_gender', data=trans_bh) ; plt.show()
sns.barplot(x='clnt_gender', y='buy_ct', hue='biz_unit', data=trans_bh) ; plt.show()

sns.barplot(x='biz_unit', y='buy_ct', hue='clnt_age', data=trans_bh) ; plt.show()
sns.barplot(x='clnt_age', y='buy_ct', hue='biz_unit', data=trans_bh) ; plt.show()

sns.catplot(x="clnt_gender", y="buy_ct", hue="clnt_age", col="biz_unit",data=trans_bh, kind="bar",height=4, aspect=.7)
sns.catplot(x="clnt_age", y="buy_ct", hue="clnt_gender", col="biz_unit",data=trans_bh, kind="bar",height=4, aspect=.7)

#%%
# 03 ------------
# 3.1 상품 코드 : 'pd_c', clac_nm1','clac_nm2', 'clac_nm3'
# ---- Y
#  01.  단순 빈도
plot_freq(trans_bh,'clac_nm1','prop',5,5)
np.around(trans_bh.groupby('clac_nm1').freq.sum() / trans_bh.freq.sum() *100,4).sort_values(ascending=False)

#  02. 구매 순서  : trans_seq', -> 내역 내 구매 순서 -> 경로적인 의미 있을 수 있음 
#  03. 구매 금액, 수량 : buy_am', 'buy_ct', 

# 3.2 muti '업종 별 상품 코드!! : biz_unit',
# ---- Y
#  01.  단순 빈도
#  02. 구매 순서  : trans_seq', -> 내역 내 구매 순서 -> 경로적인 의미 있을 수 있음 
#  03. 구매 금액, 수량 : buy_am', 'buy_ct', 

#%%
# 04 ------------
# 4.1 인적 정보 : 'clnt_gender', 'clnt_age', 
# ---- Y
#  01.  단순 빈도
plot_freq(trans_bh,'clnt_gender','prop',5,5)
plot_freq(trans_bh,'clnt_age','prop',5,5)

#  02. 구매 순서  : trans_seq', -> 내역 내 구매 순서 -> 경로적인 의미 있을 수 있음 
#  03. 구매 금액, 수량 : buy_am', 'buy_ct', 

# 4.2 muti 성별별 나이 : 'clnt_gender', 'clnt_age'
# ---- Y
#  01.  단순 빈도
#  02. 구매 순서  : trans_seq', -> 내역 내 구매 순서 -> 경로적인 의미 있을 수 있음 
#  03. 구매 금액, 수량 : buy_am', 'buy_ct', 

#%%
# 01 ------------ 

# ---- Y
#  01.  단순 빈도
#  02. 구매 순서  : trans_seq', -> 내역 내 구매 순서 -> 경로적인 의미 있을 수 있음 
#  03. 구매 금액, 수량 : buy_am', 'buy_ct', 

# 01 ------------ 
# 구매일자 : 'de_dt' - 일자별 ->  요일별 추이
# 구매 시간 : 'de_tm', - 시간대별 추이(산점도) -> 군집화로 나눠서 보기 
### new data
# *요일 : 'day_of_week'
# *주말 : 'weekend' 
# 공휴일 :

plot_freq(trans_bh,'weekend','prop',10,5)
plot_freq(trans_bh,'day_of_week','prop',10,5)