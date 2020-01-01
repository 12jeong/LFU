# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 19:24:38 2019

@author: 
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

#%% 
"""
# nobuy - kwd set 따로 
# buy / refund - 변수 붙이기
"""


