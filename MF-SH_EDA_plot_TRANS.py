# 4.1 인적 정보 : 'clnt_gender', 'clnt_age', 
# ---- Y
#  01.  단순 빈도
plot_freq(trans_bh,'clnt_gender','prop',5,5)
plot_freq(trans_bh,'clnt_age','prop',5,5)

# 2. '업종 단위 : biz_unit',
# ---- Y
#  01.  단순 빈도
plot_freq(trans_bh,'biz_unit','prop',5,5)

#  03. 구매 금액, 수량 : buy_am', 'buy_ct', 
# -- 구매 금액

# -- 구매 수량
plot_freq(trans_bh,'buy_ct','prop',10,5)

trans_bh['buy_ct'].unique()
np.around(trans_bh.groupby('buy_ct').freq.sum() / trans_bh.freq.sum() *100,4).sort_values(ascending=False)

A01_trans = trans_bh[trans_bh['biz_unit']=='A01']
A02_trans = trans_bh[trans_bh['biz_unit']=='A02']
A03_trans = trans_bh[trans_bh['biz_unit']=='A03']


# 01 ------------ 
# 구매일자 : 'de_dt' - 일자별 ->  요일별 추이
# 구매 시간 : 'de_tm', - 시간대별 추이(산점도) -> 군집화로 나눠서 보기 
### new data
# *요일 : 'day_of_week'
# *주말 : 'weekend' 
# 공휴일 :

plot_freq(trans_bh,'weekend','prop',10,5)
plot_freq(trans_bh,'day_of_week','prop',10,5)




#%% ==============================================================

# -- 구매 금액 

# ------------ 층화 : biz_unit, 나이, 성별
sns.catplot(x="clnt_gender", y="buy_am", hue="clnt_age", col="biz_unit",data=trans_bh, kind="bar",height=4, aspect=.7)
sns.catplot(x="clnt_age", y="buy_am", hue="clnt_gender", col="biz_unit",data=trans_bh, kind="bar",height=4, aspect=.7)

sns.FacetGrid(trans_bh, col="biz_unit", height=4, aspect=.5).map(sns.barplot, "clnt_age", "buy_am");
sns.FacetGrid(trans_bh, col="biz_unit", height=4, aspect=.5).map(sns.barplot, "clnt_gender", "buy_am");


sns.catplot(x="biz_unit", y="buy_am", kind="box", data=trans_bh, showfliers=False)
sns.boxplot(x="biz_unit", y="buy_am", hue="clnt_gender", data=trans_bh, palette="Set1", showfliers=False)
sns.boxplot(x="biz_unit", y="buy_am", hue="clnt_age", data=trans_bh, palette="Set1", showfliers=False)

#sns.FacetGrid(trans_bh, col="biz_unit", height=4, aspect=.5).map(sns.boxplot, "clnt_gender", "buy_am", showfliers=False);

sns.catplot(x="clnt_gender", y="buy_am", kind="box", data=trans_bh, showfliers=False)
sns.boxplot(x="clnt_gender", y="buy_am", hue="biz_unit", data=trans_bh, palette="Set1", showfliers=False)

sns.catplot(x="clnt_age", y="buy_am", kind="box", data=trans_bh, showfliers=False)
sns.boxplot(x="clnt_age", y="buy_am", hue="biz_unit", data=trans_bh, palette="Set1", showfliers=False)


# -- 구매 수량

sns.barplot(x='biz_unit', y='buy_ct', hue='clnt_gender', data=trans_bh) ; plt.show()
sns.barplot(x='clnt_gender', y='buy_ct', hue='biz_unit', data=trans_bh) ; plt.show()

sns.barplot(x='biz_unit', y='buy_ct', hue='clnt_age', data=trans_bh) ; plt.show()
sns.barplot(x='clnt_age', y='buy_ct', hue='biz_unit', data=trans_bh) ; plt.show()

sns.catplot(x="clnt_gender", y="buy_ct", hue="clnt_age", col="biz_unit",data=trans_bh, kind="bar",height=4, aspect=.7)
sns.catplot(x="clnt_age", y="buy_ct", hue="clnt_gender", col="biz_unit",data=trans_bh, kind="bar",height=4, aspect=.7)


# == = 구매 수량
# -- 날짜 : 'weekend' , 
sns.barplot(x='weekend', y='buy_ct', hue='biz_unit', data=trans_bh) ; plt.show()
sns.catplot(x="weekend", y="buy_ct", hue="clnt_age", col="biz_unit",data=trans_bh, kind="bar",height=4, aspect=.7)
sns.catplot(x="weekend", y="buy_ct", hue="clnt_gender", col="biz_unit",data=trans_bh, kind="bar",height=4, aspect=.7)


# -- 날짜 : 'day_of_week'
sns.barplot(x='day_of_week', y='buy_ct', hue='biz_unit', data=trans_bh) ; plt.show()
# A01_trans = trans_bh[trans_bh['biz_unit']=='A01']
# A02_trans = trans_bh[trans_bh['biz_unit']=='A02']
# A03_trans = trans_bh[trans_bh['biz_unit']=='A03']

sns.catplot(x="day_of_week", y="buy_ct", hue="clnt_age", col="biz_unit",data=trans_bh, kind="bar",height=4, aspect=.7)
sns.barplot(x='day_of_week', y='buy_ct', hue='clnt_age', data=A01_trans) ; plt.show()
sns.barplot(x='day_of_week', y='buy_ct', hue='clnt_age', data=A02_trans) ; plt.show()
sns.barplot(x='day_of_week', y='buy_ct', hue='clnt_age', data=A03_trans) ; plt.show()
sns.catplot(x="day_of_week", y="buy_ct", hue="clnt_gender", col="biz_unit",data=trans_bh, kind="bar",height=4, aspect=.7)
