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

#%% kwdclac3
trans_info.shape
temp = pd.merge(df_clac3,trans_info[['clnt_id', 'trans_id', 'de_dt','clac_nm3']], how='inner',
                           left_on =['clnt_id', 'sess_dt','trans_id'], 
                           right_on =['clnt_id', 'de_dt','trans_id'])
temp = temp.drop(['de_dt'],axis=1)
kwd_clac3 = temp[~temp.craw_clac3.isnull()]
dtm3_list = kwd_clac3[['craw_clac3','clac_nm3']]

n = len(dtm3_list.craw_clac3.unique()) # 844 - KWD CLAC
m = len(dtm3_list.clac_nm3.unique()) # 593 - PROD CLAC
DTM3 = dtm3_list.groupby(['craw_clac3', 'clac_nm3'])['clac_nm3'].count().unstack().fillna(0) # DTM = pd.crosstab(dtm_list.craw_clac2, dtm_list.clac_nm2)
# DTM3.to_csv(pdir+"\\Dropbox\\LFY\\datasets\\crawling\\DTM_clac3.csv",index=False)
DTM3_array = np.genfromtxt(pdir+"\\Dropbox\\LFY\\datasets\\crawling\\DTM_clac3.csv",dtype=(int,int),delimiter=',',skip_header=1)

#%% LDA utils function - collapsed Gibbs sampling
#%%
import operator  
import lda
import time
import random

def print_top_words_for_topics(vocab, topics, beta, counts=None, n_words=10):
    
    print("FORMAT |")
    print("Topic(Counts) : List of the top %d words"%n_words)
    print("-----------------------------------------------------------------------------------")
    for topic, count in zip(topics, counts):
        idx = np.argsort(beta[topic,:])[::-1]
        print('Topic {} ({}): {}'.format(topic, count,', '.join(operator.itemgetter(*idx[:n_words])(vocab))))

            
def top_topics_of_document(n, gamma, n_topics=None,option=False):
    idx = np.argsort(gamma[n,:])[::-1]

    print("Top %d topics of document %d"%(n_topics,n))
    print("-----------------------------------------------------------------------------------")

    top_topics = idx[:n_topics]
    prob = gamma[n,idx[:n_topics]]
    
    for t,p in zip(top_topics,prob):
        print("topic %d : %f"%(t,p))


def ppselect(X, T_list, n_iter=1000):
        
    random.seed(433)
    perplexity = [] 
    for n_t in T_list:
        ldamodel = lda.LDA(n_t, n_iter, random_state=1)
        ldamodel.fit(DTM_array)
        ll = ldamodel.loglikelihood()
        perplexity.append(np.exp(-ll/ X.sum()))

    # -- plot
    mpl.rcParams.update({'font.size':14})
    plt.rc('font',family='Malgun Gothic') # windows
    plt.rcParams['axes.facecolor'] = 'white'
    plt.plot(T_list,perplexity,label='perplexity', linestyle='-', lw = 1, marker='.', color='#ff6600')
    plt.style.use('fast')
    plt.grid(color='gray', ls = '-', lw = 0.25)
    plt.xlabel('Number of Topics')
    plt.ylabel('Perplexity')
    plt.show()
    
    # -- pirnt optimal number of topics    
    opt_ntopic = perplexity.index(min(perplexity))+1
    print('The Optimal number of topics is {}'.format(opt_ntopic))
    
    return perplexity , opt_ntopic

#%% SLECT THE NUMBER OF TOPICS
start_time = time.time() 
max_K = 20
T_list = np.arange(1,max_K+1,1)
perplexity , opt_ntopic = ppselect(DTM3_array, T_list, n_iter=10000)
opt_ntopic
end_time = time.time()  
print("WorkingTime: {} sec".format(int(end_time-start_time)))

#%% LDA -FITTING
opt_ntopic
model = lda.LDA(n_topics=3, n_iter=1000, random_state=1)
model.fit(DTM3_array)
model.loglikelihood()

beta = model.topic_word_ # beta
gamma = model.doc_topic_ # gamma

vocab_list = DTM3.columns.tolist()
document =  DTM3.index

import pickle
with open(pdir+"\\Dropbox\\LFY\\datasets\\LDApickle\\lda_clac3.pickle", "wb") as f:
   pickle.dump((beta, gamma, vocab_list, document, perplexity , opt_ntopic ), f)
#with open(pdir+"\\Dropbox\\LFY\\datasets\\LDApickle\\lda_clac3.pickle", "rb") as f:
#    beta, gamma, vocab_list, document, perplexity , opt_ntopic   = pickle.load(f) 

#%% OUTPUT 1 [ topic - terms ] 
assigned_topics = np.argmax(beta, axis=0)
counts = np.bincount(assigned_topics)
topid_idx = np.argsort(counts)[::-1]

print_top_words_for_topics(vocab_list ,topid_idx , beta , counts=counts[topid_idx] , n_words=5)

#%% OUTPUT 2 [ doc - topic] 
which_doc = 8 ; top_ntopics = 5

top_topics_of_document(which_doc, gamma, top_ntopics, option=True)


    










