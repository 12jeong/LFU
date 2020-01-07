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
pcsh = "C:\\Users\\YongTaek\\"
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
#%% Example for LDA ___ #  https://pypi.org/project/lda/
import numpy as np
import lda

docnames =  ['doc1', 'doc4', 'doc3', 'doc2']
vocab = ['science', 'mining', 'c', 'text', 'nlp', 'structures',
   'processing', 'matrix', 'r', 'algorithms', 'data',
   'programming', 'python', 'cleaning']
dtm = np.array([[0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 2, 0, 1, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0],
                [1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 1]])
  
model = lda.LDA(n_topics=3, n_iter=1000, random_state=1)
model.fit(dtm)

topic_word = model.topic_word_
n_top_words = 3

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    
#%% DATA LOAD
df_clac2   = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\crawling\\df_clac2.csv") 
df_clac3   = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\crawling\\df_clac3.csv") 
trans_info = pd.read_csv(pdir+"\\Dropbox\\LFY\\datasets\\ppdata\\trans_info.csv")
online_bh  = pd.read_csv(pdir+"\\Dropbox\\LFY\\datasets\\ppdata\\online_bh.csv")
df_buy     = pd.read_csv(pdir+"\\Dropbox\\LFY\\datasets\\ppdata\\df_buy.csv")
df_nobuy   = pd.read_csv(pdir+"\\Dropbox\\LFY\\datasets\\ppdata\\df_nobuy.csv")
#%% df_buy + crawling kwd
kwd_clac2 = pd.merge(df_clac2[['clnt_id','sess_dt','sess_id','hit_seq','craw_kwd','craw_clac2']],df_buy, how='right',on =  ['clnt_id','sess_dt','sess_id','hit_seq']).drop_duplicates()

# kwdclac2
temp1 = kwd_clac2[kwd_clac2.trans_id.isna()]
temp2 = kwd_clac2[~kwd_clac2.trans_id.isna()]

temp3 = pd.merge(kwd_clac2,trans_info[['clnt_id', 'trans_id', 'de_dt','clac_nm2']], how='inner', # 1072445 / 575606 
                           left_on =['clnt_id', 'sess_dt','trans_id'], 
                           right_on =['clnt_id', 'de_dt','trans_id']).drop_duplicates() 

#%%  ppdata 모두 병합 - online_bh['action_type'] == 6인 행으로 구성된 데이터 셋 
ppdata = df_buy[df_buy['action_type'] == 6][['action_type', 'clnt_id','sess_dt', 'sess_id','trans_id',  # 식별자 
                                              'hit_seq', 'hit_tm', 'hit_pss_tm', # 조회 정보                                                       
                                              'tot_pag_view_ct','tot_sess_hr_v', # 다른 action_type내 동일 - 추가 정제 필요 X
                                              'biz_unit','trfc_src', 'dvc_ctg_nm', 'clnt_gender', 'clnt_age', # 층화 변수
                                              'id']].drop_duplicates() # 중복 없음

# 01. 키워드 리스트 : df_kwd['kwdlist'] | 병합 -> kwd_ppdata
df_kwd = df_buy[df_buy['action_type']==0].groupby(['clnt_id','sess_dt','sess_id','trans_id'])['sech_kwd'].apply(list).reset_index()
df_kwd.columns = ['clnt_id', 'sess_dt', 'sess_id', 'trans_id', 'kwd_list']
df_kwd['kwd_list'] = df_kwd['kwd_list'].apply(', '.join)

df_kwd['kwd'] = 1 # kwd list 존재 유무 변수 - > nan : 구매를 위한 검색 존재 X
kwd_ppdata = pd.merge(ppdata,df_kwd, how='left', on =['clnt_id', 'sess_dt', 'sess_id', 'trans_id']).drop_duplicates()

# unique_kwd : 키워드 unique 목록 저장
unique_kwd = df_buy['sech_kwd'].unique()
with open(pdir+"\\Dropbox\\LFY\\datasets\\ppdata\\unique_kwd.txt",'w', encoding='utf-8') as f:
    for item in unique_kwd:
        f.write("%s\n" % item)