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
#%% DATA LOAD
df_clac2   = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\crawling\\df_clac2.csv") 
df_clac3   = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\crawling\\df_clac3.csv") 
trans_info = pd.read_csv(pdir+"\\Dropbox\\LFY\\datasets\\ppdata\\trans_info.csv")
online_bh  = pd.read_csv(pdir+"\\Dropbox\\LFY\\datasets\\ppdata\\online_bh.csv")
df_buy     = pd.read_csv(pdir+"\\Dropbox\\LFY\\datasets\\ppdata\\df_buy.csv")
df_nobuy   = pd.read_csv(pdir+"\\Dropbox\\LFY\\datasets\\ppdata\\df_nobuy.csv")

#%% kwdclac2
trans_info.shape
temp = pd.merge(df_clac2,trans_info[['clnt_id', 'trans_id', 'de_dt','clac_nm2']], how='inner', # 1072445 / 575606 
                           left_on =['clnt_id', 'sess_dt','trans_id'], 
                           right_on =['clnt_id', 'de_dt','trans_id'])
temp = temp.drop(['de_dt'],axis=1)
kwd_clac2 = temp[~temp.craw_clac2.isnull()]
dtm_list = kwd_clac2[['craw_clac2','clac_nm2']]

n = len(dtm_list.craw_clac2.unique()) #288 - KWD CLAC
m = len(dtm_list.clac_nm2.unique()) # 181 - PROD CLAC
DTM = dtm_list.groupby(['craw_clac2', 'clac_nm2'])['clac_nm2'].count().unstack().fillna(0) # DTM = pd.crosstab(dtm_list.craw_clac2, dtm_list.clac_nm2)
# DTM.to_csv(pdir+"\\Dropbox\\LFY\\datasets\\crawling\\DTM_clac2.csv",index=False)
DTM_array = np.genfromtxt(pdir+"\\Dropbox\\LFY\\datasets\\crawling\\DTM_clac2.csv",dtype=(int,int),delimiter=',',skip_header=1)

# 확인
DTM.loc["Gimbap Sushi Salad"].loc["Fish Cakes and Crab Sticks"]
DTM.loc["1 small household food"].loc["Yogurt"]
dtm_list[(dtm_list.craw_clac2=="Gimbap Sushi Salad") & (dtm_list.clac_nm2=="Fish Cakes and Crab Sticks")].shape
# 노의미
plt.figure(figsize=(10,50))
sns.heatmap(DTM.describe()[1:].transpose(),
            annot=True,linecolor="w",
            linewidth=2,cmap=sns.color_palette("muted"))
plt.title("Data summary")
plt.show()

#%% LDA
import lda

model = lda.LDA(n_topics=5, n_iter=1000, random_state=1)
model.fit(DTM_array)

topic_word = model.topic_word_
n_top_words = 5

vocab =  DTM.columns

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' , '.join(topic_words)))
    
