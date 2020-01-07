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

from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from selenium.webdriver.common.keys import Keys

df_buy = pd.read_csv(r"C:\Users\YongTaek\Dropbox\LFY\datasets\ppdata\df_buy.csv")
# 1. 마트에 관련된 키워드만 넣는다. 
kwd_dic = df_buy[df_buy.biz_unit=='A03'].sech_kwd.unique()
len(kwd_dic)
kwd_dic = list(kwd_dic)
kwd_dic.remove(np.nan)
len(kwd_dic)

# 사전 정의
dic = {}
err_kwd = []
dic

# 드라이브 설정
options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument("disable-gpu")
driver = webdriver.Chrome(r'C:\Users\YongTaek\Desktop\chromedriver_win32\chromedriver',chrome_options=options)
driver.implicitly_wait()

for i in kwd_dic[12467:]:
    # 롯데마트에 진입
    driver.get('http://www.lottemart.com')
    driver.implicitly_wait(50)

    # 키워드 검색
    div_elem = driver.find_element_by_class_name('swindow')
    elem = div_elem.find_element_by_id("searchTerm")
    elem.send_keys(i)
    elem.submit()
    
    check = driver.find_element_by_class_name('wrap-even-crstab')
    check_txt = check.find_element_by_css_selector('p').text
    if ((check_txt.split(' ')[-1]=='없습니다.')|(check_txt.split(' ')[-1]=='상품입니다.')|(check_txt.split(' ')[-1]=='않습니다.')|(check_txt.split(' ')[-1]=='담으세요.')):
        err_kwd.append(i)
        print('err_kwd추가완료',kwd_dic.index(i))
    else :
        key = driver.find_element_by_partial_link_text(check_txt)
        driver.implicitly_wait(50)
        key.send_keys(Keys.ENTER)
        driver.implicitly_wait(50)
        
        try:
            driver.switch_to.alert.accept()
        except:
            pass

        try:
            error = driver.find_element_by_id("container").find_element_by_css_selector('p').text.split(' ')[-1]
        except:
            error = '죄송합니다.'

        if (error=='죄송합니다.'):
            err_kwd.append(i)
            print('err_kwd추가완료',kwd_dic.index(i))
        else:
            kwd_cls = driver.find_element_by_class_name("wrap-location")
            id_list = kwd_cls.find_elements_by_xpath("//div[@id]")
            string = str()
            for k in range(len(id_list)):
                if('divCat' in id_list[k].get_attribute("id")):
                    string = string + id_list[k].text + str('///')
            dic[i]=string
            print('dic추가완료',kwd_dic.index(i))




len(dic)+len(err_kwd)
kwd_dic.index(i)
kwd_dic[12466]
dic['시크릿데이']
check_txt
id_list

kwd_dic.index(i)
err_kwd.append(kwd_dic[4849]) #미미네 다른사이트 
err_kwd.append(kwd_dic[4925]) #시크릿 쥬쥬 다른사이트
err_kwd.append(kwd_dic[6983]) #물총 다른사이트
err_kwd.append(kwd_dic[7645]) #헬로카봇 다른사이트
err_kwd.append(kwd_dic[8509]) #스마트픽
err_kwd.append(kwd_dic[12090]) #롯데마트
err_kwd.append(kwd_dic[12466]) #드라이브 픽
# 딕셔너리 저장.
import pickle
# save
with open(r'C:\Users\YongTaek\Desktop\lotte_dic.pickle', 'wb') as f:
    pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)

# 리스트 저장
with open('err_kwd.pickle', 'wb') as f:
    pickle.dump(err_kwd, f)

# load
with open(r'C:\Users\YongTaek\Desktop\lotte_dic.pickle', 'rb') as f:
    lotte = pickle.load(f)

# 리스트 불러오기
with open(r'C:\Users\YongTaek\Desktop\err_kwd.pickle', 'rb') as f:
    lotte_err = pickle.load(f)
lotte_err

