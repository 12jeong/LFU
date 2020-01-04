# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------------------------
@ Created on Fri Jan  3 16:25:17 2020
---------------------------------------------------------------------------------------
@ Author: SHyun46
---------------------------------------------------------------------------------------
@ Code descript 
---------------------------------------------------------------------------------------
  Goal     : Biz_unit [A03, B03] - 특성 파악 및 고객 유형 제안
  Contents : 
---------------------------------------------------------------------------------------
"""
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
import pickle
mpl.rcParams.update({'font.size':14})
plt.rc('font',family='Malgun Gothic') # windows
%matplotlib inline 

#%% expand output display pd.df
pd.set_option('display.expand_frame_repr', False) 
#%% load
with open(pc+'Dropbox\LFY\datasets\crawling\lotte_dic.pickle', 'rb') as f:
    lotte = pickle.load(f)
with open(pc+'Dropbox\LFY\datasets\crawling\err_kwd.pickle', 'rb') as f:
    lotte_err = pickle.load(f)
with open(pc+'Dropbox\LFY\datasets\crawling\prepro_dic.pickle', 'rb') as f:
    prepro_dic = pickle.load(f)
# -- clnc
with open(pc+'Dropbox\LFY\datasets\crawling\clnc1.pickle', 'rb') as f:
    clnc1 = pickle.load(f)
with open(pc+'Dropbox\LFY\datasets\crawling\clnc2.pickle', 'rb') as f:
    clnc2 = pickle.load(f)
with open(pc+'Dropbox\LFY\datasets\crawling\clnc3.pickle', 'rb') as f:
    clnc3 = pickle.load(f)
    
online_bh      = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\rawdata\\online_bh.csv")     
trans_info      = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\rawdata\\trans_info.csv")  
    
#%%
lotte # kwd -> 대중소
lotte_err # 검색 결과 안나오는거 
prepro_dic
clnc1
clnc2
clnc3




    