import pandas as pd
import pickle
import numpy as numpy

df_buy = pd.read_csv(r"C:\Users\YongTaek\Dropbox\LFY\datasets\ppdata\df_buy.csv")
trans_info = pd.read_csv(r"C:\Users\YongTaek\Dropbox\LFY\datasets\ppdata\trans_info.csv")
df_buy.columns
df_buy.sech_kwd

with open(r'C:\Users\YongTaek\Desktop\realfin_dic.pickle', 'rb') as f:
    lotte = pickle.load(f)
with open(r'C:\Users\YongTaek\Desktop\fin_dic.pickle', 'rb') as f:
    clnc3 = pickle.load(f)


with open(r'C:\Users\YongTaek\Desktop\yong_cl1.pickle', 'rb') as f:
    clnc1 = pickle.load(f)    


lotte

key_lst = []
val_lst = []
for i,j in lotte.items():
    key_lst.append(i)
    val_lst.append(j)
    print(i+ str(':')+j)

key_lst = []
val_lst = []
for i,j in clnc3.items():
    key_lst.append(i)
    val_lst.append(j)
    print(i+ str(':')+j)

key_lst = []
val_lst = []
for i,j in clnc1.items():
    key_lst.append(i)
    val_lst.append(j)
    print(i+ str(':')+j)

lotte_dic = pd.DataFrame([key_lst, val_lst],index=None).T
clnc3_dic = pd.DataFrame([key_lst, val_lst],index=None).T
clnc1_dic = pd.DataFrame([key_lst, val_lst],index=None).T

lotte_dic.columns = ['kwd', 'Class']
clnc3_dic.columns = ['kwd', 'Class']
clnc1_dic.columns = ['kwd', 'Class']
lotte_dic
clnc3_dic

# 일단 df_buy에 중분류 머지.
merge = df_buy.merge(lotte_dic, left_on='sech_kwd', right_on='kwd',how='left')

small = df_buy.merge(clnc3_dic, left_on='sech_kwd', right_on='kwd', how='left')

big = df_buy.merge(clnc1_dic, left_on='sech_kwd', right_on='kwd', how='left')
len(small.clnt_id.unique())
small[small.clnt_id==2].head(60)

col = ['clnt_id', 'sess_dt', 'sess_id', 'action_type', 'kwd', 'kwd1', 'Class', 'trans_id']
small[col]

small['kwd1'] = small.groupby(['clnt_id','sess_dt','sess_id'])['sech_kwd'].ffill()
big['kwd1'] = big.groupby(['clnt_id','sess_dt','sess_id'])['sech_kwd'].ffill()

len(trans_info.trans_id.value_counts())
trans_A = trans_info[trans_info.biz_unit.isin(['A01','A02','A03'])]
trans_A.trans_id.value_counts()[trans_A.trans_id.value_counts()>1]

trans_A[trans_A.trans_id==51589].tail(50)
trans_A[trans_A.trans_id==102723].tail(50)
trans_A[trans_A.trans_id==59981].tail(50)
trans_A[trans_A.trans_id==82525].tail(50)
# Only price 문제 처리해야함.

trans_A
lotte_dic[lotte_dic.Class=='Only Price']

big.to_csv(r'C:\Users\YongTaek\Desktop\big.csv')