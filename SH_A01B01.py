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

mg_ppdata      = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\ppdata\\mg_ppdata.csv")   #  51297
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

#%% 범주형 - 층화 변수
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

