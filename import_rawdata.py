#%% import
%reset -f
import os
os.getcwd()
from os import chdir

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

pd.set_option('display.expand_frame_repr', False) # expand output display pd.df
#%%
data1 = pd.read_csv("C:\\Users\\MYCOM\\Dropbox\\LFU\\제6회 L.POINT Big Data Competition-분석용데이터-01.온라인 행동 정보.csv",low_memory=False) 
data2 = pd.read_csv("C:\\Users\\MYCOM\\Dropbox\\LFU\\제6회 L.POINT Big Data Competition-분석용데이터-02.거래 정보.csv") 
data3 = pd.read_csv("C:\\Users\\MYCOM\\Dropbox\\LFU\\제6회 L.POINT Big Data Competition-분석용데이터-03.고객 Demographic 정보.csv") 
data4 = pd.read_csv("C:\\Users\\MYCOM\\Dropbox\\LFU\\제6회 L.POINT Big Data Competition-분석용데이터-04.상품분류 정보.csv") 



