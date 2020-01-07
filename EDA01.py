%reset -f
# import
import os
os.getcwd()
from os import chdir
#os.chdir('C:\\Users\\UOS\\Documents\\GITHUB\LFU')
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

pd.set_option('display.expand_frame_repr', False) # expand output display pd.df

# Load raw data
raw_online_bh  = pd.read_csv("C:\\Users\\YongTaek\\Dropbox\\LFY\\datasets\\rawdata\\online_bh.csv")   #  3196362
raw_trans_info = pd.read_csv("C:\\Users\\YongTaek\\Dropbox\\LFY\\datasets\\rawdata\\trans_info.csv") 
raw_demo_info  = pd.read_csv("C:\\Users\\YongTaek\\Dropbox\\LFY\\datasets\\rawdata\\demo_info.csv") 
raw_prod_info  = pd.read_csv("C:\\Users\\YongTaek\\Dropbox\\LFY\\datasets\\rawdata\\prod_info.csv") 

# copy raw data
online_bh = raw_online_bh.copy()
trans_info= raw_trans_info.copy()
demo_info = raw_demo_info.copy()
prod_info = raw_prod_info.copy()

# data1 & data3 merge
df_right = [raw_demo_info]
for dtidx in range(len(df_right)):
    dt_temp = df_right[dtidx].copy()
    online_bh = online_bh.merge(dt_temp, on='clnt_id', how='left')

# 2,3,4 merge
prod_info_uk=raw_prod_info.copy()
prod_info_uk['pd_c'] = prod_info_uk['pd_c'].astype(str).str.zfill(4)
prod_info_uk.loc[1668]=['unknown','unknown','unknown','unknown']

trans_info = raw_trans_info.copy()
df_right = [raw_demo_info,prod_info_uk]
key = ['clnt_id','pd_c']
how = ['left','left']
for dtidx in range(len(df_right)):
    dt_temp = df_right[dtidx].copy().drop_duplicates()
    trans_info = trans_info.merge(dt_temp, on=key[dtidx], how=how[dtidx])

# data1 : online_bh 전처리 
online_bh.head()
online_bh["id"] = online_bh.index
online_bh["hit_pss_tm"] = online_bh["hit_pss_tm"]/(60*1000)   # 분으로 변환
online_bh["tot_sess_hr_v"] = online_bh["tot_sess_hr_v"]/60    # 분으로 변환

# action_type == 6 or 7 인데 trans_id == NaN 인 행 제거
sum(np.isnan(online_bh[(online_bh.action_type == 6) | (online_bh.action_type == 7) ].trans_id)) 
online_bh = online_bh[~( (np.isnan(online_bh.trans_id) == True) & ((online_bh.action_type==6)| (online_bh.action_type == 7)))]
sum(np.isnan(online_bh[(online_bh.action_type == 6) | (online_bh.action_type == 7) ].trans_id)) 

## action_type == 7 & trans_id 중복된 행, 첫 행만 보존
df_action7 =online_bh.copy()
df_action7 = df_action7[df_action7['action_type']==7]
df_action7.sort_values(['clnt_id','sess_id','sess_dt','hit_seq']).groupby(['clnt_id','sess_id','sess_dt']).head()
df_action7_nd = df_action7.drop_duplicates(['trans_id'],keep="first")
df_action7_nd.sort_values(['clnt_id','sess_id','sess_dt','hit_seq']).groupby(['clnt_id','sess_id','sess_dt']).head()
df_action_n7 =online_bh.copy()
df_action_n7 = df_action_n7[df_action_n7['action_type']!=7]
online_bh=pd.concat([df_action7_nd, df_action_n7]) 


# nobuy 정의 : 동일 session내에서 action_type = 6(구매 완료) 없는 사람 
df_buy =online_bh.copy()
df_buy = df_buy[df_buy['action_type']==6]

buy_session_key = df_buy[['clnt_id','sess_id','sess_dt']]
buy_session_key['buy']=1
buy_session_key

temp = online_bh.copy()
temp=temp.merge(buy_session_key, how='left').drop_duplicates() # 중복행 제거해야함

temp.head()

df_nobuy =temp[temp['buy'].isna()]
df_buy = temp[temp['buy'].notnull()]

df_nobuy.shape[0] + df_buy.shape[0] - online_bh.shape[0] # 행 개수 확인

# buy 정의 : 같은 session 내에서 구매후 환불을 하지 않은 사람 (노이즈 제거) - 추후 환불 고객 분석 가능

buy_tmp = df_buy.copy()
buy_tmp2 = buy_tmp.groupby(['clnt_id','sess_id','sess_dt','trans_id'])['id'].agg(trans_count='count') < 2
no_cancel_key = buy_tmp2[ buy_tmp2['trans_count'].notna()].reset_index()[['clnt_id','sess_id','sess_dt']]
buy_tmp   = buy_tmp.merge(no_cancel_key, how='inner').drop_duplicates() 
sum(buy_tmp.groupby(['clnt_id','sess_id','sess_dt','trans_id'])['id'].agg('count') > 1) # 이 경우는 같은세션에서 다른 상품을 구매한 것
df_buy = buy_tmp


''' df_buy에 검색 키워드와 구매 매칭 작업 '''

# buy데이터에서 유용할것 같은 컬럼만 get
df_buy.columns
df_buy_tmp = df_buy

# 그룹별 시간에 따라 정렬한 데이터. sort_buy_tmp
# 먼저 sort하고 그다음 groupby하면 빠름.
sort_buy_tmp = df_buy_tmp.groupby(['clnt_id','sess_dt','sess_id']).apply(lambda x: x.sort_values(by = ['hit_seq'], ascending = True))
sort_buy_tmp.index
sort_buy_tmp.index.names = ['clnt_id', 'sess_dt', 'sess_id','org_ind']
sort_buy_tmp = sort_buy_tmp.reset_index(level=[1])
sort_buy_tmp.index = pd.Index(range(len(sort_buy_tmp)))
sort_buy_tmp['sess_dt'] = sort_buy_tmp['sess_dt'].astype('str')
sort_buy_tmp.info()
sort_buy_tmp

# action_type에 검색에만 검색 키워드 존재.
for i in range(7):
    print("%d 액션타입의 검색키워드 결측치 수 %d" %(i, sort_buy_tmp[sort_buy_tmp.action_type==i].sech_kwd.isnull().sum()))
for i in range(7):
    print("%d 액션타입의 원래 갯수 %d" %(i, len(sort_buy_tmp[sort_buy_tmp.action_type==i])))

# action_type이 엄청 많은사람이랑 너무 짧은사람이랑 무슨 차이일까
sort_buy_tmp.clnt_id.unique()[:20]
sort_buy_tmp[sort_buy_tmp.clnt_id==2].head(50)
trans_info[trans_info.trans_id=='62037']
sort_buy_tmp[sort_buy_tmp.clnt_id==4]
trans_info[trans_info.trans_id=='85046']
sort_buy_tmp[sort_buy_tmp.clnt_id==9]
trans_info[trans_info.trans_id=='45242']
sort_buy_tmp[sort_buy_tmp.clnt_id==12]
trans_info[trans_info.trans_id=='54099']
sort_buy_tmp[sort_buy_tmp.clnt_id==17]
trans_info[trans_info.trans_id=='64325']
sort_buy_tmp[sort_buy_tmp.clnt_id==19]
trans_info[trans_info.trans_id=='72882']
sort_buy_tmp[sort_buy_tmp.clnt_id==20]
trans_info[trans_info.trans_id=='64923']
sort_buy_tmp[sort_buy_tmp.clnt_id==22]
trans_info[trans_info.trans_id=='102568']
sort_buy_tmp[sort_buy_tmp.clnt_id==23]
sort_buy_tmp[sort_buy_tmp.clnt_id==24]

# trans_info에서 biz_unit == B인 행 날리기
raw_trans_info = raw_trans_info[raw_trans_info.biz_unit.isin(['A01', 'A02', 'A03'])]
trans_info = trans_info[trans_info.biz_unit.isin(['A01', 'A02', 'A03'])]
trans_info['de_dt'] = trans_info['de_dt'].astype('str')
raw_trans_info

# 오직 구매목록만 빼냄
record_buy = sort_buy_tmp[sort_buy_tmp.action_type==6]

# 구매를 한 고객에 대하여 trans_id는 결측이 없음.
sort_buy_tmp[sort_buy_tmp.action_type==6].trans_id.isnull()

# trans_id에 중복 값이 있음.
sort_buy_tmp[sort_buy_tmp.action_type==6].trans_id.value_counts()

'''
# 중복 값 중에 trans_id가 89064인거 확인 => 다 같은 고객.
record_buy[record_buy.trans_id==89064]
record_buy[record_buy.trans_id==61030]
record_buy[record_buy.trans_id==38744]
record_buy[record_buy.trans_id==66779]
record_buy[record_buy.trans_id==38730]


# 물품정보에는 없다.
trans_info[trans_info.trans_id==89064]
trans_info[trans_info.trans_id==61030]
trans_info[trans_info.trans_id==38744]
trans_info[trans_info.trans_id==66779]
trans_info[trans_info.trans_id==38730]
'''

# 사전 행동유형 없이 한 번에 에어컨을 삼. (데이터 어떻게 처리할 것인지.)
record_buy[record_buy.trans_id=='59330']
df_buy[df_buy.clnt_id==14245]
sort_buy_tmp[sort_buy_tmp.trans_id==59330] # 에어컨 한번에 산거 아님.
trans_info[trans_info.trans_id==59330]

# 구매를 한 고객의 trans_id와 상품 목록의 trans_id의 unique 개수가 다름.
# 구매 고객 trans_id
len(sort_buy_tmp[sort_buy_tmp.action_type==6].trans_id.unique())
len(trans_info.trans_id.unique())
record_buy.index

len(record_buy.merge(trans_info, on=['clnt_id', 'trans_id','de_dt','de_tm'], how='left'))


# merge가 잘 안돼서 action_type==6에서 valuecounts()하여 trans_id의 빈도가 가장높은
# 89064를 보니 다 같은 구매자. 
record_buy.trans_id.value_counts()
record_buy[record_buy.trans_id==89064]
trans_info[trans_info.trans_id==89064]
trans_info[trans_info.clnt_id==53496]
df_buy[df_buy.trans_id==89064]


# 구매 기록엔 하나뜨는데 거래 기록에는 5개가 나옴.
record_buy[(record_buy.clnt_id==39423) & (record_buy.trans_id==69372)].head(5)
trans_info[(trans_info.clnt_id==39423) & (trans_info.trans_id==69372)].head(5)

record_buy[(record_buy.clnt_id==39423)& (record_buy.trans_id==69407)]
trans_info[(trans_info.clnt_id==39423) & (trans_info.trans_id==69407)]

# 구매 기록에 같은 사람이 많이 뜨는 사람들도 trans_id가 두개인 사람은 없음.
record_buy.clnt_id.value_counts()
np.sum(record_buy[(record_buy.clnt_id==45279)].trans_id.value_counts()>1)
np.sum(record_buy[(record_buy.clnt_id==68146)].trans_id.value_counts()>1)
# 즉 같은 구매자라도 trans_id는 고유하다.
record_buy[(record_buy.clnt_id==45279)]
trans_info[trans_info.clnt_id==45279]

## 조인할 때 날짜와 시간까지 같은걸로 해서 join
trans_info.clnt_id.value_counts()
trans_info[(trans_info.clnt_id==42575) & (trans_info.de_dt==20190722)]
record_buy[record_buy.clnt_id==42575]

trans_info[['de_dt','de_tm']]
trans_info.de_tm.value_counts()
trans_info

# sess_dt, hit_tm이 sess_dt, hit_tm 이랑 같은 것 같은지?
# 같다면 그걸로 조인 해보자

record_buy = record_buy.rename(columns={"sess_dt":"de_dt", "hit_tm":"de_tm"})
record_buy.merge(trans_info, on=['clnt_id', 'trans_id','de_dt','de_tm'], how='left')

# clnt_id, trans_id, de_dt, de_tm가 겹치는 것도 있음 하...
trans_info['clnt_id'] = trans_info['clnt_id'].astype(str)
trans_info['trans_id'] = trans_info['trans_id'].astype(str)
tmp = trans_info.clnt_id + str('_')+trans_info.trans_id + str('_') +trans_info.de_dt + str('_')+ trans_info.de_tm
# trans_info 중에 겹치는거.
tmp.value_counts()

trans_info.info()
record_buy.info()
record_buy['clnt_id'] = record_buy['clnt_id'].astype('str')
record_buy['trans_id'] = record_buy['trans_id'].astype('int')
record_buy['trans_id'] = record_buy['trans_id'].astype('str')

tmp2 = record_buy.clnt_id + str('_') + record_buy.trans_id + str('_') + record_buy.de_dt + str('_')+ record_buy.de_tm


# 10002_47347 이걸보면 분명 0710 18:05에 구입했다는 action이 나오는데
# 구매 정보에서는 18시:05에 많이 샀다고 나옴.
# 장바구니에 넣고 많이 산것.?? 
trans_info[(trans_info.clnt_id=='10002') & (trans_info.trans_id=='47347')&
(trans_info.de_dt=='20190710')]


# hit_pss_tm 내림차순 clnt_id 그룹 dedt detm 
# record_buy 하나 선택해서 trans_info를 기준으로 left 조인.
for i in tmp2.value_counts().head(16).index:
    x,y,j,k = i.split('_')
    temp = record_buy[(record_buy.clnt_id==x)&(record_buy.trans_id==y)&
    (record_buy.de_dt==j)&(record_buy.de_tm==k)]
    print(np.argmin(temp.hit_pss_tm))
    
    #record_buy = record_buy.drop(index=np.argmin(temp.hit_pss_tm),axis=0)

''' 잘 빠졌나 확인
record_buy[(record_buy.clnt_id=='51913')&(record_buy.trans_id=='92916')&
(record_buy.de_dt=='20190904')&(record_buy.de_tm=='15:48')]

record_buy[(record_buy.clnt_id=='14608')&(record_buy.trans_id=='74516')&
(record_buy.de_dt=='20190813')&(record_buy.de_tm=='16:40')]

record_buy[(record_buy.clnt_id=='51913')&(record_buy.trans_id=='50827')&
(record_buy.de_dt=='20190715')&(record_buy.de_tm=='13:39')]

record_buy[(record_buy.clnt_id=='40610')&(record_buy.trans_id=='91036')&
(record_buy.de_dt=='20190903')&(record_buy.de_tm=='18:42')]

record_buy[(record_buy.clnt_id=='54850')&(record_buy.trans_id=='110996')&
(record_buy.de_dt=='20190927')&(record_buy.de_tm=='14:32')]

record_buy[(record_buy.clnt_id=='30421')&(record_buy.trans_id=='61030')&
(record_buy.de_dt=='20190927')&(record_buy.de_tm=='14:32')]

record_buy[(record_buy.clnt_id=='54850')&(record_buy.trans_id=='110996')&
(record_buy.de_dt=='20190927')&(record_buy.de_tm=='14:32')]

record_buy[(record_buy.clnt_id=='54850')&(record_buy.trans_id=='110996')&
(record_buy.de_dt=='20190927')&(record_buy.de_tm=='14:32')]
'''

# 키로 잘 구성되어 있는지 확인
tmp2 = record_buy.clnt_id + str('_') + record_buy.trans_id + str('_') + record_buy.de_dt + str('_')+ record_buy.de_tm


# merge 성공
trans_join = trans_info.merge(record_buy, on=['clnt_id', 'trans_id', 'de_dt','de_tm'],how='left')
trans_join[trans_join.org_ind.notnull()]
len(trans_join.clac_nm1.value_counts())


# action type이 6인것만 골라내서 unique 한지 본 뒤에 merge하기
sort_buy_tmp[sort_buy_tmp.clnt_id==2].head(50)
trans_join[trans_join.trans_id=='62037']
trans_join.action_type

# action_type이 6인 것만 뽑는게 trans_info에도 있고 buy의 trans_id도 같이 있는거임.
# tmp3는 품목에도 있고 구매자가 산 기록도 있는거.
trans_ac6 = trans_join[trans_join.action_type.notnull()]
trans_ac6.action_type = trans_ac6.action_type.astype('str')
trans_ac6.hit_pss_tm = trans_ac6.hit_pss_tm.astype('str')

# 위에는 action_type이 6인것만 뽑았는데 이것을 보면 꼭 그렇게 하면 안될수 도 있음.
trans_join[trans_join.clnt_id=='58265'].head(19)
sort_buy_tmp[sort_buy_tmp.clnt_id==58265]

# tmp3에서 clnt_id, trans_id, de_dt, de_tm 유니크한 것보다 작은행만 추출??
trans_ac6
trans_ac6[trans_ac6.duplicated(['clnt_id','trans_id','de_dt','de_tm'])]
tmptrans_ac6.head()
keyword = sort_buy_tmp[sort_buy_tmp.sech_kwd.notnull()]


tmp = keyword[keyword.clnt_id==2]
tmp
sort_buy_tmp[sort_buy_tmp.clnt_id==2].head(50)
trans_info[trans_info.clnt_id=='2'].sort_values(by='de_dt')
trans_join.columns

sort_buy_tmp


# hit_pss_tm은 누적이라 일단 각 세션에 있는시간으로 분류.
df1 = df_buy
df1['hit_diff'] = df1.sort_values(['clnt_id','sess_dt','sess_id','hit_seq']).groupby(['clnt_id','sess_id','sess_dt'])['hit_pss_tm'].diff(periods=-1)*-1
df1.sort_values(['clnt_id','sess_dt','sess_id','hit_seq'])[['clnt_id','hit_pss_tm','hit_diff']]

sort_buy_tmp = df1.groupby(['clnt_id','sess_dt','sess_id']).apply(lambda x: x.sort_values(by = ['hit_seq'], ascending = True))

sort_buy_tmp.info()

# kwd를 세션에 따라서 분류.
tmp = [sort_buy_tmp.index[x][3] for x in range(len(sort_buy_tmp))]
len(tmp)
sort_buy_tmp = sort_buy_tmp.reset_index(drop=True)
sort_buy_tmp['org_ind'] = pd.DataFrame(tmp)
sort_buy_tmp.head(50)

sort_buy_tmp['kwd'] = sort_buy_tmp.groupby(['clnt_id','sess_dt','sess_id'])['sech_kwd'].ffill()
sort_buy_tmp.kwd.isnull().sum()
sort_buy_tmp.kwd2.isnull().sum()



sort_buy_tmp.groupby(['clnt_id','sess_dt','kwd2'])['hit_diff'].sum()

sort_buy_tmp
trans_join