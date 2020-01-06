# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------------------------
@ Created on Wed Jan  1 22:22:30 2020
---------------------------------------------------------------------------------------
@ Author: SHyun46
---------------------------------------------------------------------------------------
@ Code descript 
---------------------------------------------------------------------------------------
  Goal     : Biz_unit [A03, B03] - 특성 파악 및 고객 유형 제안
  Contents : 
---------------------------------------------------------------------------------------
"""
#% A03 
#- 검색키워드 존재 하는 사람이 많음
#- 여자 많
#- 40대 -> 30대 -> 50대 -> 20,60 대 -> 10대 (거의 없) | unknown
#- 가중치 곱해줘도 평일이 1.59배 많음 : 온라인이라 별 상관 없는 듯 = (79.87*2) / (20.13*5)
#- 키워드 존재 하는 사람이 많음 2/3
#- direct -> push
#- mobile web ->pc
#
## -- 검색 개수 별 행태 | 
#- 성별차이 없
#- 60대 잘 안 찾 vs 30,20,10대 많이 찾
#- 주말에 좀더 검색 많
#- 웹사이트 조금쓰 검색 ** 이건 좀 주요 포인트일들 
#- 웹웹 많 **
#
## -- 구매 상품 분류 개수별 행태 -  unique를 해야할지.... -> 생각점 해바야
#- 여자 상품 분류 기록 개수 많음
#- 60대 고객 적지만 상품 많이삼
#- [known] 10대 검색은 많이 하지만 구매 중 상품분류찍힌건 적음
#- [unknown] 상품 분류 한개도 업ㄱ음 / 전부 unknown
#- [unknown] 웹사이트 조금쓰 검색 ** 이건 좀 주요 포인트일들 (추이좀 다름)


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
#%%-- DATA FIGURE
pd.set_option('display.expand_frame_repr', False)  # expand output display pd.df
#%% LOAD DATASET
online_bh      = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\ppdata\\online_bh.csv")     
trans_info      = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\ppdata\\trans_info.csv")  

df_buy         = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\ppdata\\df_buy.csv")   
df_nobuy       = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\ppdata\\df_nobuy.csv")  
df_design_buy  = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\ppdata\\df_design_buy.csv",index_col=0) 

raw_mg_ppdata      = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\ppdata\\mg_ppdata.csv")   #  51297
#%% FUNCTION FOR PLOT
# -- 단순 빈도
def plot_freq(data,var,ytype,xsize,ysize) :
    dataset = data.copy()
    dataset['freq'] = 1
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
    plt.title("Frequncies of "+ var ,fontsize= 16)
    plt.xlabel('Type of '+ var, fontsize=15)
    if ytype == "freq":
        plt.ylabel('Frequency of '+ var, fontsize=15)
    elif ytype == "prop":
        plt.ylabel('Proportion of '+ var, fontsize=15)
    
    for i, rect in enumerate(plot):       
        ax.text(rect.get_x() + rect.get_width() / 1.3, 0.95 * rect.get_height(), str(round(y[i],2)) + '%', ha='right', va='center')
        
#%% GOAL1 | Biz_unit [A03] 특성 파악 : df_design_buy


#%% 범주형 - 층화 변수
# -- 요일 추가
mg_ppdata = raw_mg_ppdata.copy()
mg_ppdata.columns
mg_ppdata['sess_dt']
mg_ppdata['sess_dt'] = pd.to_datetime(mg_ppdata['sess_dt'])
mg_ppdata['day_of_week'] = mg_ppdata['sess_dt'].dt.day_name()
mg_ppdata['weekend'] = 0
mg_ppdata['weekend'][mg_ppdata.day_of_week.isin(['Saturday','Sunday'])] = 1
mg_ppdata.kwd[~mg_ppdata.kwd.isna()] = 'kwd'
mg_ppdata.kwd[mg_ppdata.kwd.isna()] = 'no kwd'
mg_ppdata['clnt'] = 'clnt'
mg_ppdata['clnt'][mg_ppdata.clnt_gender=='unknown'] = 'no clnt'
mg_ppdata['clnt'].unique()
#mg_ppdata.kwd = mg_ppdata.kwd.astype('str')
type(mg_ppdata.kwd_list[0])

# -- 검색 키워드 개수 : nkwd 
mg_ppdata.nkwd = mg_ppdata.kwd_list.str.split(',').str.len()
mg_ppdata.nkwd[mg_ppdata.nkwd.isna()] = 0 
mg_ppdata.nkwd = mg_ppdata.nkwd.astype(int)

# 키워드 5단위 범주
mg_ppdata['nkwdB'] = 0
mg_ppdata['nkwdB'][~mg_ppdata.nkwd.isin([0])] = mg_ppdata.nkwd//5+1
mg_ppdata['nkwdB'] = mg_ppdata['nkwB'].astype(int)

# -- 구매 상품 분류 개수 : nkwd 
mg_ppdata['nclac'] = mg_ppdata.clac_nm1.str.split(',').str.len()
mg_ppdata['nclac'][mg_ppdata['nclac'].isna()] = 0 
mg_ppdata['nclac'] = mg_ppdata.nclac.astype(int)

# 구매 상품 분류 5단위 범주
mg_ppdata['nclacB'] = 0
mg_ppdata['nclacB'][~mg_ppdata.nclac.isin([0])] = mg_ppdata.nclac//5+1
mg_ppdata['nclacB'] = mg_ppdata['nclacB'].astype(int)
#%%
mg_ppdata.biz_unit.unique()
mg_A03 = mg_ppdata[mg_ppdata.biz_unit=='A03']
mg_A03_noclnt = mg_A03[mg_A03.clnt_gender=='unknown']
mg_A03_clnt = mg_A03[mg_A03.clnt_gender!='unknown']
#%% *** 전체 | 나이/성별/주말평일 별로 구매품목 / 키워드 보기
DF = mg_A03
plot_freq(DF,'clnt','prop',5,5) # 키워드 존재 하는 사람이 많음
plot_freq(DF,'clnt_gender','prop',5,5) # 여자 많
plot_freq(DF,'clnt_age','prop',5,5) # 40대 -> 30대 -> 50대 -> 20,60 대 -> 10대 (거의 없) | unknown
plot_freq(DF,'weekend','prop',5,5) # 가중치 곱해줘도 평일이 1.59배 많음 : 온라인이라 별 상관 없는 듯 = (79.87*2) / (20.13*5)
plot_freq(DF,'kwd','prop',5,5) # 키워드 존재 하는 사람이 많음 2/3
plot_freq(DF,'trfc_src','prop',10,5) # direct -> push
plot_freq(DF,'dvc_ctg_nm','prop',10,5) # mobile web ->pc

# -- 검색 개수 별 행태 | 
plot_freq(DF,'nkwd','prop',10,5) 
plot_freq(DF,'nkwdB','prop',10,5) 
sns.catplot( y="nkwd", kind="box", data=DF)
sns.catplot(x= 'clnt_gender' , y="nkwd", kind="box", data=DF, showfliers=False) # 차이 없
sns.catplot(x= 'clnt_age' , y="nkwd", kind="box", data=DF, showfliers=False) # 60대 잘 안 찾 vs 30,20,10대 많이 찾
sns.catplot(x= 'weekend' , y="nkwd", kind="box", data=DF, showfliers=False) # 주말에 좀더 검색 많
sns.catplot(x= 'trfc_src' , y="nkwd", kind="box", data=DF, showfliers=False) # 웹사이트 조금쓰 검색 ** 이건 좀 주요 포인트일들 
sns.catplot(x= 'dvc_ctg_nm' , y="nkwd", kind="box", data=DF , showfliers=False) # 웹웹 **

# -- 구매 상품 분류 개수별 행태   -  unique를 해야할지....  -> 생각 점
plot_freq(DF,'nclac','prop',10,5) 
plot_freq(DF,'nclacB','prop',10,5) 
sns.catplot( y="nclac", kind="box", data=DF)
sns.catplot(x= 'clnt_gender' , y="nclac", kind="box", data=DF, showfliers=False) # 여자 많이 삼
sns.catplot(x= 'clnt_age' , y="nclac", kind="box", data=DF, showfliers=False) # 60대 고객 적지만 상품 많이삼
sns.catplot(x= 'weekend' , y="nclac", kind="box", data=DF, showfliers=False) # 같
sns.catplot(x= 'trfc_src' , y="nclac", kind="box", data=DF, showfliers=False) # \
sns.catplot(x= 'dvc_ctg_nm' , y="nclac", kind="box", data=DF , showfliers=False) # \

#%% ***  known 고객 | mg_A03_clnt
DF = mg_A03_clnt
plot_freq(DF,'clnt_gender','prop',5,5) # [같] 여자 많
plot_freq(DF,'clnt_age','prop',5,5) # [같] 40대 -> 30대 -> 50대 -> 20,60 대 -> 10대 (거의 없) | unknown
plot_freq(DF,'weekend','prop',5,5) # [같] 가중치 곱해줘도 평일이 1.59배 많음 : 온라인이라 별 상관 없는 듯 
plot_freq(DF,'kwd','prop',5,5) # [같] 키워드 존재 하는 사람이 많음 2/3
plot_freq(DF,'trfc_src','prop',10,5) # [같] direct -> push
plot_freq(DF,'dvc_ctg_nm','prop',10,5) # [같]  mobile web ->pc

# -- 검색 개수 별 행태 | 
plot_freq(DF,'nkwd','prop',10,5)  # [같] 
plot_freq(DF,'nkwdB','prop',10,5) # [같] 
sns.catplot( y="nkwd", kind="box", data=DF)# [같] 
sns.catplot(x= 'clnt_gender' , y="nkwd", kind="box", data=DF, showfliers=False) # [같] 여자 많
sns.catplot(x= 'clnt_age' , y="nkwd", kind="box", data=DF, showfliers=False) # [같]  60대 잘 안 찾 vs 30,20,10대 많이 찾
sns.catplot(x= 'weekend' , y="nkwd", kind="box", data=DF, showfliers=False) # [같] 주말에 좀더 검색 많
sns.catplot(x= 'trfc_src' , y="nkwd", kind="box", data=DF, showfliers=False) # 웹사이트 조금쓰 검색 ** 이건 좀 주요 포인트일들 
sns.catplot(x= 'dvc_ctg_nm' , y="nkwd", kind="box", data=DF , showfliers=False) # 웹웹 **

# -- 구매 상품 분류 개수별 행태   -  unique를 해야할지....  -> 생각 점
plot_freq(DF,'nclac','prop',10,5) 
plot_freq(DF,'nclacB','prop',10,5) 
sns.catplot( y="nclac", kind="box", data=DF)
sns.catplot(x= 'clnt_gender' , y="nclac", kind="box", data=DF, showfliers=False) # [같]여자 많이 삼
sns.catplot(x= 'clnt_age' , y="nclac", kind="box", data=DF, showfliers=False) # 10대 검색은 많이 하지만 구매 중 상품분류찍힌건 적음
sns.catplot(x= 'weekend' , y="nclac", kind="box", data=DF, showfliers=False)# [같] 차이 없 -skewed
sns.catplot(x= 'trfc_src' , y="nclac", kind="box", data=DF, showfliers=False) # [같]
sns.catplot(x= 'dvc_ctg_nm' , y="nclac", kind="box", data=DF , showfliers=False) # [같] 웹

#%% *** unknown 고객 | mg_A03_noclnt
DF = mg_A03_noclnt
plot_freq(DF,'weekend','prop',5,5) # [겉]전체랑 다를게 없 : 가중치 곱해줘도 평일이 1.57배 많음 : 온라인이라 별 상관 없는 듯 = (79.77*2) / (20.23*5)
plot_freq(DF,'kwd','prop',5,5) # 키워드 존재 하는 사람이 많음
plot_freq(DF,'clac_nm1','prop',5,5) # 상품 분류 한개도 업ㄱ음
plot_freq(DF,'trfc_src','prop',10,5) # [같] direct -> push [다] direct가 더 비중 많아짐, push 줄고
plot_freq(DF,'dvc_ctg_nm','prop',10,5) # [같]  mobile web ->pc [다] pc -> 더 비중 많아짐 ,mobile web 줄고

# -- 검색 개수 별 행태 | 
plot_freq(DF,'nkwd','prop',10,5)  # [같] 
plot_freq(DF,'nkwdB','prop',10,5) # [같] 
sns.catplot( y="nkwd", kind="box", data=DF)# [같] 
sns.catplot(x= 'weekend' , y="nkwd", kind="box", data=DF, showfliers=False) # [같] 주말에 좀더 검색 많
sns.catplot(x= 'trfc_src' , y="nkwd", kind="box", data=DF, showfliers=False) # 웹사이트 조금쓰 검색 ** 이건 좀 주요 포인트일들 (추이좀 다름)
sns.catplot(x= 'dvc_ctg_nm' , y="nkwd", kind="box", data=DF , showfliers=False) # [같] 웹웹 **

# -- 구매 상품 분류 개수별 행태   # 전부 unknown
#%% trans_info
trans_info.columns


#%%
# -- 온라인 - A03
# plot_freq(mg_ppdata,'biz_unit','prop',5,5) # 세션별 구매행에 대한 업종별 차지 비율
plot_freq(df_design_buy,'biz_unit','prop',5,5)
df_A03 = df_design_buy[df_design_buy.biz_unit == 'A03']
col_order = ['clnt_id', 'sess_dt','sess_id', # 분류 변수 
             'biz_unit',  'tot_pag_view_ct','tot_sess_hr_v', 'trfc_src', 'dvc_ctg_nm', 'clnt_gender', 'clnt_age', # 기타 변수
             'id', 'buy',
             'action_count_0', 'action_count_1', 'action_count_2', 'action_count_3',
               'action_count_4', 'action_count_5', 'action_count_6', 'action_count_7',
               'action_time_0', 'action_time_1', 'action_time_2', 'action_time_3',
               'action_time_4', 'action_time_5', 'action_time_6', 'action_time_7',
               'tot_seq',] # 추가 보조 변수
df_A03=df_A03[col_order]
df_A03 = df_A03.sort_values(['clnt_id', 'sess_dt','sess_id'],axis=0) # 오름차순 int int int int object
dataset =  df_A03

plot_freq(dataset,'trfc_src','prop',10,5)
plot_freq(dataset,'dvc_ctg_nm','prop',10,5)
plot_freq(dataset,'clnt_gender','prop',10,5)
plot_freq(dataset,'clnt_age','prop',10,5)
plot_freq(dataset ,'buy','prop',10,5)

# -- trfc_src
sns.catplot(x="trfc_src", y="action_count_0", kind="box", data=dataset, showfliers=False)
sns.boxplot(x="trfc_src", y="action_count_0", hue="dvc_ctg_nm", data=dataset, palette="Set1", showfliers=False)
sns.boxplot(x="trfc_src", y="action_count_0", hue="clnt_gender", data=dataset, palette="Set1", showfliers=False)
sns.boxplot(x="trfc_src", y="action_count_0", hue="clnt_age", data=dataset, palette="Set1", showfliers=False)
sns.boxplot(x="trfc_src", y="action_count_0", hue="buy", data=dataset, palette="Set1", showfliers=False)

# -- dvc_ctg_nm
sns.catplot(x="dvc_ctg_nm", y="action_count_0", kind="box", data=dataset, showfliers=False)
sns.boxplot(x="dvc_ctg_nm", y="action_count_0", hue="trfc_src", data=dataset, palette="Set1", showfliers=False)
sns.boxplot(x="dvc_ctg_nm", y="action_count_0", hue="clnt_gender", data=dataset, palette="Set1", showfliers=False)
sns.boxplot(x="dvc_ctg_nm", y="action_count_0", hue="clnt_age", data=dataset, palette="Set1", showfliers=False)
sns.boxplot(x="dvc_ctg_nm", y="action_count_0", hue="buy", data=dataset, palette="Set1", showfliers=False)

# -- clnt_gender
sns.catplot(x="clnt_gender", y="action_count_0", kind="box", data=dataset, showfliers=False)
sns.boxplot(x="clnt_gender", y="action_count_0", hue="trfc_src", data=dataset, palette="Set1", showfliers=False)
sns.boxplot(x="clnt_gender", y="action_count_0", hue="dvc_ctg_nm", data=dataset, palette="Set1", showfliers=False)
sns.boxplot(x="clnt_gender", y="action_count_0", hue="clnt_age", data=dataset, palette="Set1", showfliers=False)
sns.boxplot(x="clnt_gender", y="action_count_0", hue="buy", data=dataset, palette="Set1", showfliers=False)

# -- clnt_age
sns.catplot(x="clnt_age", y="action_count_0", kind="box", data=dataset, showfliers=False)
sns.boxplot(x="clnt_age", y="action_count_0", hue="trfc_src", data=dataset, palette="Set1", showfliers=False)
sns.boxplot(x="clnt_age", y="action_count_0", hue="dvc_ctg_nm", data=dataset, palette="Set1", showfliers=False)
sns.boxplot(x="clnt_age", y="action_count_0", hue="clnt_gender", data=dataset, palette="Set1", showfliers=False)
sns.boxplot(x="clnt_age", y="action_count_0", hue="buy", data=dataset, palette="Set1", showfliers=False)

# -- buy
sns.catplot(x="buy", y="action_count_0", kind="box", data=dataset, showfliers=False)
sns.boxplot(x="buy", y="action_count_0", hue="trfc_src", data=dataset, palette="Set1", showfliers=False)
sns.boxplot(x="buy", y="action_count_0", hue="dvc_ctg_nm", data=dataset, palette="Set1", showfliers=False)
sns.boxplot(x="buy", y="action_count_0", hue="clnt_gender", data=dataset, palette="Set1", showfliers=False)
sns.boxplot(x="buy", y="action_count_0", hue="clnt_age", data=dataset, palette="Set1", showfliers=False)

#%% 연속형 

#%% GOAL2 | Biz_unit [A03] - 고객 유형 제안

#%%
# -- 오프라인 B03
plot_freq(trans_info,'biz_unit','prop',5,5) # 구매 구성 행이 많다

