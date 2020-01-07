import pickle
import pandas as pd
with open(r'C:\Users\YongTaek\Desktop\real_dic2.pickle', 'rb') as f:
    lotte = pickle.load(f)
with open(r'C:\Users\YongTaek\Desktop\err_kwd.pickle', 'rb') as f:
    lotte_err = pickle.load(f)
with open(r'C:\Users\YongTaek\Desktop\clnc1.pickle', 'rb') as f:
    clnc1 = pickle.load(f)
with open(r'C:\Users\YongTaek\Desktop\clnc2.pickle', 'rb') as f:
    clnc2 = pickle.load(f)
with open(r'C:\Users\YongTaek\Desktop\clnc3.pickle', 'rb') as f:
    clnc3 = pickle.load(f)


with open(r'C:\Users\YongTaek\Desktop\real_dic2.pickle', 'rb') as f:
    real_fin = pickle.load(f)
len(real_fin)


# 에러키워드 처리

dic_prepro = {}
# 뒤에 필요없는 기호 전처리
for i,j in real_fin.items():
    print(i, j)
    if j[-3:] == '///':
        string = j[0:-3]
        dic_prepro[i] = string
        print('뒤에 삭제')
    else:
        dic_prepro[i] = j
        print('그대로')
with open(r'C:\Users\YongTaek\Desktop\real_dic3.pickle', 'wb') as f:
    pickle.dump(dic_prepro, f, pickle.HIGHEST_PROTOCOL)
with open(r'C:\Users\YongTaek\Desktop\real_dic2.pickle', 'rb') as f:
    lotte = pickle.load(f)

# 분류군 기호 바꾸기
dic_prepro1 = {}
for i,j in dic_prepro.items():
    dic_prepro1[i] = j.replace("///","*") 



# 저장
with open(r'C:\Users\YongTaek\Desktop\prepro_dic.pickle', 'wb') as f:
    pickle.dump(dic_prepro1, f, pickle.HIGHEST_PROTOCOL)


dic_prepro={}

# 분류군의 띄어쓰기 전처리
for i,j in dic_prepro1.items():
    dic_prepro[i] = j.replace("·"," ")
# 분류군의 띄어쓰기 전처리
for i,j in dic_prepro.items():
    dic_prepro1[i] = j.replace("   "," ")
    # 분류 나뉘는 기호 전처리
for i,j in dic_prepro1.items():
    dic_prepro[i] = j.replace("*","/") 

# 저장
with open(r'C:\Users\YongTaek\Desktop\real_prepro.pickle', 'wb') as f:
    pickle.dump(dic_prepro, f, pickle.HIGHEST_PROTOCOL)

with open(r'C:\Users\YongTaek\Desktop\real_prepro.pickle', 'rb') as f:
    lotte = pickle.load(f)

    
len(lotte)
# 번역기 셀레니움
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import numpy as np
import time

# 번역기 접근
driver = webdriver.Chrome(r'C:\Users\YongTaek\Desktop\chromedriver_win32\chromedriver')
driver.get('https://translate.google.co.kr/?hl=ko')

# 번역기 입력창 클리어
driver.find_element_by_id('source').clear()

# 딕셔너리 순서를 모르기 때문에 리스트로 접근


key_lst = []
val_lst = []
for i,j in lotte.items():
    key_lst.append(i)
    val_lst.append(j)
    print(i+ str(':')+j)

lotte_dic = pd.DataFrame([key_lst, val_lst],index=None).T
np.sum(lotte_dic.iloc[:,1]=='')

fin_dic={}
driver.find_element_by_id('source').clear()

err_ind = []
err1_ind = []
err2_ind = []
#err1_ind
for i in err1_ind:
    '''
    if len(lotte_dic.iloc[i,1].split('/'))==1:
        string = lotte_dic.iloc[i,1].split('/')[0]
    else:
        string = lotte_dic.iloc[i,1].split('/')[-2]
    '''
    string = lotte_dic.iloc[i,1].split('/')[0]
    elem = driver.find_element_by_id("source")
    elem.send_keys(string)
    time.sleep(2)

    result = driver.find_elements_by_css_selector("span[class='tlid-translation translation']")
    if result != []:
        try:
            print(i,str(','),result[0].text)
            fin_dic[lotte_dic.iloc[i,0]] = result[0].text
        except:
            err2_ind.append(i)
    elif result ==[]:
        err2_ind.append(i)
    driver.find_element_by_id('source').clear()
i
len(fin_dic)+len(err2_ind)





lotte_dic.iloc[err_ind,:]
key_lst[3893]
val_lst[3893]
err_ind

# 저장
with open(r'C:\Users\YongTaek\Desktop\yong_cl1.pickle', 'wb') as f:
    pickle.dump(fin_dic, f, pickle.HIGHEST_PROTOCOL)