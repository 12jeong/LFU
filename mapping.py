import pandas as pd
import pickle
import numpy as np

df_buy = pd.read_csv(r"C:\Users\YongTaek\Dropbox\LFY\datasets\ppdata\df_buy.csv")
trans_info = pd.read_csv(r"C:\Users\YongTaek\Dropbox\LFY\datasets\ppdata\trans_info.csv")

df_buy['kwd1'] = df_buy.groupby(['clnt_id','sess_dt','sess_id'])['sech_kwd'].ffill()

df_buy[['clnt_id','sess_dt','sess_id','hit_seq','action_type','sech_kwd','kwd1','trans_id']].head(60)

dir(trans_info.groupby(['trans_id']))

# 소분류 대분류 중분류 모두 사용할거임.
tr_id_key3 = trans_info.groupby(['trans_id'])['clac_nm3'].apply(lambda x: list(x))
tr_id_key2 = trans_info.groupby(['trans_id'])['clac_nm2'].apply(lambda x: list(x))
tr_id_key1 = trans_info.groupby(['trans_id'])['clac_nm1'].apply(lambda x: list(x))

# 세개를 합쳐서 키로.
tr_id_key = pd.DataFrame([tr_id_key1,tr_id_key2,tr_id_key3]).T
tr_id_key = tr_id_key.reset_index()
small = pd.read_csv(r'C:\Users\YongTaek\Dropbox\LFY\datasets\Crawling\small.csv')
mid = pd.read_csv(r'C:\Users\YongTaek\Dropbox\LFY\datasets\Crawling\middle.csv')
big = pd.read_csv(r'C:\Users\YongTaek\Desktop\big.csv')
mid.columns
inner = inner.iloc[:, 1:]
inner[inner.action_type==6]


# Class (내가 크롤링한 키워드) ffil해야함 그룹바이해서 아이디 세션 날짜별로 
mid['fillkwd'] = mid.groupby(['clnt_id', 'sess_dt', 'sess_id']).Class.ffill()
small['fillkwd'] = small.groupby(['clnt_id', 'sess_dt', 'sess_id']).Class.ffill()
big['fillkwd'] = big.groupby(['clnt_id', 'sess_dt', 'sess_id']).Class.ffill()

inner1 = mid.merge(tr_id_key, on='trans_id', how='inner')
inner2 = small.merge(tr_id_key, on='trans_id', how='inner')
inner3 = big.merge(tr_id_key, on='trans_id', how='inner')
## trans_info에도 있고, df_buy에도 있는 것만 저장
inner_ac6_1 = inner1[inner.action_type==6]
inner_ac6_2 = inner2[inner.action_type==6]
inner_ac6_3 = inner3[inner.action_type==6]

# 분모개수 10440개 그 세션에서 키워드가 있고 구매도 한사람.
kwd_buyer1 = inner_ac6_1[inner_ac6.fillkwd.notnull()]
kwd_buyer2 = inner_ac6_2[inner_ac6.fillkwd.notnull()]
kwd_buyer3 = inner_ac6_3[inner_ac6.fillkwd.notnull()]
len(kwd_buyer)

# 내가 크롤링한 clac kwd 전처리
lst = [x for x in kwd_buyer1.fillkwd.astype(str)]
lst = [x.lower() for x in lst]
lst = [x.split(' ') for x in lst]
lst1 = []
for i in range(len(lst)):
    lst1.append(list(pd.DataFrame(lst[i]).iloc[:,0].unique()))
len(lst1)

lst = [x for x in kwd_buyer2.fillkwd.astype(str)]
lst = [x.lower() for x in lst]
lst = [x.split(' ') for x in lst]
lst2 = []
for i in range(len(lst)):
    lst2.append(list(pd.DataFrame(lst[i]).iloc[:,0].unique()))
len(lst2)

lst = [x for x in kwd_buyer3.fillkwd.astype(str)]
lst = [x.lower() for x in lst]
lst = [x.split(' ') for x in lst]
lst3 = []
for i in range(len(lst)):
    lst3.append(list(pd.DataFrame(lst[i]).iloc[:,0].unique()))
len(lst3)

# 중분류(lst1) 과 소분류(lst2)의 키워드합침.
lst123 = [lst1[x] + lst2[x] +lst3[x] for x in range(len(lst1))]
lst1[0]
lst2[0]
lst3[0]
lst123[0]
len(lst123)


# 원래 소분류의 clac 전처리
clac_lst1 =  kwd_buyer.clac_nm1
clac_lst1 = [' '.join(x) for x in clac_lst1]
clac_lst1 = [x.replace('/', '') for x in clac_lst1]
clac_lst1 = [x.replace('-', '') for x in clac_lst1]
clac_lst1 = [x.lower() for x in clac_lst1]
clac_lst1 = [x.split(' ') for x in clac_lst1]
# 중분류 전처리
clac_lst2 =  kwd_buyer.clac_nm2
clac_lst2 = [' '.join(x) for x in clac_lst2]
clac_lst2 = [x.replace('/', '') for x in clac_lst2]
clac_lst2 = [x.replace('-', '') for x in clac_lst2]
clac_lst2 = [x.lower() for x in clac_lst2]
clac_lst2 = [x.split(' ') for x in clac_lst2]
# 대분류 전처리
clac_lst3 =  kwd_buyer.clac_nm3
clac_lst3 = [' '.join(x) for x in clac_lst3]
clac_lst3 = [x.replace('/', '') for x in clac_lst3]
clac_lst3 = [x.replace('-', '') for x in clac_lst3]
clac_lst3 = [x.lower() for x in clac_lst3]
clac_lst3 = [x.split(' ') for x in clac_lst3]

# 확률 구하기 0,2,3은 무조건 있어야함.

count = []
for i in range(len(lst123)):
    for j in lst123[i]:
        if((j in clac_lst1[i])|(j in clac_lst2[i])|(j in clac_lst3[i])):
            count.append(i)
            continue
        
len(count)

len(pd.DataFrame(count).iloc[:,0].unique()) / len(kwd_buyer)
