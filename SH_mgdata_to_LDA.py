# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------------------------
@ Created on Mon Dec 30 13:11:49 2019
---------------------------------------------------------------------------------------
@ Author: SHyun46
---------------------------------------------------------------------------------------
@ Code descript 
---------------------------------------------------------------------------------------
  Goal     : 해당 상품 분류를 구매한 고객은 어떤 단어를 검색했는지 
  Contents : DTM(문서 - 단어 행렬) 만들기 <- ** [문서]상품 분류 & [단어]검색 리스트 
---------------------------------------------------------------------------------------
"""
%reset -f
#%% import
import os
os.getcwd()
from os import chdir
#os.chdir('C:\\Users\\UOS\\Documents\\GITHUB\LFU')
pcsh = "C:\\Users\\user"
os.chdir(pcsh)
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

#%%
pd.set_option('display.expand_frame_repr', False) # expand output display pd.df

#%% Load raw data
mg_ppdata      = pd.read_csv(".\Dropbox\\LFY\\datasets\\ppdata\\mg_ppdata.csv")   #  3196362
raw_prod_info  = pd.read_csv(".\Dropbox\\LFY\\datasets\\rawdata\\prod_info.csv") 

f = open(pdir+"\\Dropbox\\LFY\\datasets\\ppdata\\unique_kwd.txt",'rt', encoding='UTF8'); txtfile = f.read() ; f.close()
unique_kwd = txtfile.strip().split('\n') ; unique_kwd = sorted(unique_kwd)
f = open(pdir+"\\Dropbox\\LFY\\datasets\\ppdata\\unique_clnc1.txt",'rt', encoding='UTF8'); txtfile = f.read() ; f.close()
unique_clnc1 = txtfile.strip().split('\n') ;unique_clnc1 = sorted(unique_clnc1)
f = open(pdir+"\\Dropbox\\LFY\\datasets\\ppdata\\unique_clnc2.txt",'rt', encoding='UTF8'); txtfile = f.read() ; f.close()
unique_clnc2 = txtfile.strip().split('\n') ;unique_clnc2 = sorted(unique_clnc2)
f = open(pdir+"\\Dropbox\\LFY\\datasets\\ppdata\\unique_clnc3.txt",'rt', encoding='UTF8'); txtfile = f.read() ; f.close()
unique_clnc3 = txtfile.strip().split('\n') ;unique_clnc3 = sorted(unique_clnc3)

#%% 문서 - 단어 행렬 만들기 <- ** [문서]상품 분류 & [단어]검색 리스트 
# 해당 상품 분류를 구매한 고객은 어떤 단어를 검색했는지 
clac_var = 'clac_nm1' # str 
kwd_list = 'kwd_list' #str

#-----------------------------------------------------------------------------------------------
# 01 ** 둘 다 존재하는 행의 차지 비율 
#-----------------------------------------------------------------------------------------------
sum(mg_ppdata[clac_var].notna() * mg_ppdata[kwd_list].notna()) / mg_ppdata.shape[0] # 0.168
sum(mg_ppdata[clac_var].notna()) / mg_ppdata.shape[0] # 0.339
sum(mg_ppdata[kwd_list].notna()) / mg_ppdata.shape[0] # 0.465

# 아래의 행을 제외하고 채널에 nan 존재하면 대/중/소 분류 모두 nan
# prod_info[pd_c,clac_nm1,clac_nm2,clac_nm3] = 196 | Chilled Foods | Packaged Side Dishes | NaN
clac = mg_ppdata[clac_var]
kwd = mg_ppdata[kwd_list]
clac_kwd =  mg_ppdata[[clac_var,kwd_list]] 
clac_m =  mg_ppdata[['clac_nm1','clac_nm2','clac_nm3']]
sum(clac_m['clac_nm1'].notna() * clac_m['clac_nm2'].notna() *clac_m['clac_nm3'].notna() ) / mg_ppdata.shape[0] 
sum(clac_m['clac_nm1'].notna() ) / mg_ppdata.shape[0] 
sum(clac_m['clac_nm2'].notna() ) / mg_ppdata.shape[0] 
sum(clac_m['clac_nm3'].notna() ) / mg_ppdata.shape[0] 
sum(clac_m['clac_nm1'].notna() * clac_m['clac_nm2'].notna()) / mg_ppdata.shape[0] 
sum(clac_m['clac_nm2'].notna() *clac_m['clac_nm3'].notna()) / mg_ppdata.shape[0] 

#%%
#-----------------------------------------------------------------------------------------------
# 02 ** DTM
#-----------------------------------------------------------------------------------------------
# [문서] 상품 분류 | clac_var = 'clac_nm1' # str 
# [단어] 검색어    | kwd_list = 'kwd_list' #str
#-----------------------------------------------------------------------------------------------
clac = mg_ppdata[clac_var]
kwd = mg_ppdata[kwd_list]

DTM_ck1 = pd.DataFrame(index=unique_clnc1,columns=unique_kwd)
DTM_ck2 = pd.DataFrame(index=unique_clnc2,columns=unique_kwd)
DTM_ck3 = pd.DataFrame(index=unique_clnc3,columns=unique_kwd)

#%% 공휴일 API님!
#%% non_buy / refund - 어케할지
#%% online_bh 내 다른 변수들 사용
#%% 추천시스템 구축




