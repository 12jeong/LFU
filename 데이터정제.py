import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 불러오기.
a = pd.read_csv(r"C:\Users\YongTaek\Desktop\엘포인트\Data\1.온라인_행동_정보.csv")
b = pd.read_csv(r"C:\Users\YongTaek\Desktop\엘포인트\Data\2.거래_정보.csv")
a.head()

# 데이터 형식 확인
a.info()

# 정수형, 실수형 컬럼 네임확인.
int_col = ['clnt_id', 'sess_id', 'hit_seq', 'action_type', 'sess_dt',
'hit_pss_tm', 'trans_id', 'tot_pag_view_ct', 'tot_sess_hr_v']
obj_col = ['biz_unit', 'hit_tm', 'trfc_src', 'dvc_ctg_nm']

# trans_id가 비정상적으로 결측치가 많음.
# trans_id : 거래 ID, 구매내역 식별용으로 중요함.
# tot_sess_hr_v : 세션 내 총 시간
# tot_pag_view_ct : 세션 내의 총 페이지(화면) 뷰 수
np.isnan(a[int_col]).sum()

# 전체의 온라인 행동 개수에서 거래를 한사람의 개수. 56989명
len(a)-np.isnan(a['trans_id']).sum()

# 구매를 한 고객과 안한고객을 나눔.
notbuy = a[a['trans_id'].isnull()]
buy = a[a['trans_id'].notnull()]

# 먼저 구매 고객의 형태
''' sech_kwd 구매한 사람의 키워드가 없다? 그리고 pdf파일에서는 srch로 표기되었음.'''
len(buy) # 총 56989건 구매.
buy.info()
buy['sech_kwd']

# 1. hit seq 조회 순서로 나열해서 클릭한 순서는 무엇인지 알아보기 action_type이용
# 비 구매자와 구매자의 tot_pag_view_ct 비교
notbuy.sort_values(by='hit_seq')[int_col]
notbuy[notbuy['clnt_id'] == 7809].sort_values(by='hit_seq')[int_col]

# 분포가 비슷하다.
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.hist(notbuy['tot_pag_view_ct'])
ax2.hist(buy['tot_pag_view_ct'])
plt.show()

# tot_sess_hr_v 의 비교. 이도 비슷하다.
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.hist(notbuy['tot_sess_hr_v'])
ax2.hist(buy['tot_sess_hr_v'])
plt.show()

# action_type : 0:검색~ 8:결제옵션 행동한 액션 취함.
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

# 구매를 한 고객들은 구매완료를 많이 한 것을 알 수 있음. 환불도 존재
ax1.hist(notbuy['action_type'])
# 구매를 하지 않은 고객은 여러개로 나눠져 있다.
ax2.hist(buy['action_type'])
plt.show()

# ** 성향을 파악하기 위해 action_type번호 별로 다시 연구하는 것이 필요함.
notbuy['action_type'].value_counts()
notbuy.groupby('action_type').count()
notbuy.groupby('action_type')[['hit_pss_tm']].mean()

# - action_type별 검색 키워드를 살펴봐야할 것 같음.
# 그룹에 따라 한번에 나타나게 하니까 깨지는 것이 있음.
notbuy['sech_kwd'].isnull().sum()
notbuy.groupby('action_type')['sech_kwd'].value_counts()

# action_type : 0(검색)에 해당하는 키워드는 결측치 없음. 
notbuy[notbuy['action_type']==0]['sech_kwd'].value_counts()
# action_type : 1(제품목록)에 해당하는 키워드는 전부 결측값임.
len(notbuy[notbuy['action_type']==1])
notbuy[notbuy['action_type']==1]['sech_kwd'].isnull().sum()
# action_type : 2(제품 세부정보 보기)에 해당하는 키워드는 전부 결측값임.
len(notbuy[notbuy['action_type']==2])
notbuy[notbuy['action_type']==2]['sech_kwd'].isnull().sum()
# action_type : 3(장바구니)에 해당하는 키워드도 전부 결측
len(notbuy[notbuy['action_type']==3])
notbuy[notbuy['action_type']==3]['sech_kwd'].isnull().sum()
# action_type : 4(장바구니 제품 삭제)에 해당하는 키워드도 전부 결측
len(notbuy[notbuy['action_type']==4])
notbuy[notbuy['action_type']==4]['sech_kwd'].isnull().sum()
# action_type : 5(결제 시도)에 해당하는 키워드도 전부 결측
len(notbuy[notbuy['action_type']==5])
notbuy[notbuy['action_type']==5]['sech_kwd'].isnull().sum()
# action_type : 6(구매 완료)에 해당하는 키워드도 전부 결측
len(notbuy[notbuy['action_type']==6])
notbuy[notbuy['action_type']==6]['sech_kwd'].isnull().sum()
# 비구매자 중 action_type : 7(구매 환불)에 해당하는 사람은 없음.
len(notbuy[notbuy['action_type']==7])
# 비구매자 중 action_type : 8(결제 옵션)에 해당하는 사람은 없음.
len(notbuy[notbuy['action_type']==8])

# 비구매자 들은 음식을 대체로 많이 검색을한다. 
''' 이 비구매자들의 아이디랑 구매자 아이디 비교해봐야함 '''
''' 구매자 비구매자가 나눠져 있지 않은상태에서 구매한아이디의 검색내용 다불러와서 검색순서 및 패턴 분석해야함.'''
tmp = notbuy[notbuy['action_type']==0]['sech_kwd'].value_counts()
len(tmp[tmp>30])
tmp

# 2. biz_unit업종 단위 오프라인인지 온라인인지도 나누어 보아야할 것 같음.
# 3. hit_pss_tm 조회 경과시간이 얼마나 긴지도 알아보아야함. 
# 4. sech_kwd 검색 키워드 체크., 기기 유형 체크
# 5. dvc_ctg_nm : 기기 유형 기기유형별로 구매 비구매 차이 봐야함.

a[a['clnt_id'] == 7809].sort_values(by='hit_seq')

b.groupby(['clnt_id']).apply(lambda x: x.sort_values(by = 'de_tm', ascending = False))





# 2019-12-13
a[a['action_type']==7].sum()


# 이정쓰 innerjoin
tmp = a[a['trans_id'].notnull() & (a.action_type==6)]
key = tmp[['clnt_id','sess_id','sess_dt']].drop_duplicates()
merge = pd.merge(a,key,how='inner')
merge
len(a[~a['id'].isin(merge['id'])])+len(merge)

len(a)
