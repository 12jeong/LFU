import time
# 오류 키워드 다시 크롤링
with open(r'C:\Users\YongTaek\Desktop\real_dic1.pickle', 'rb') as f:
    lotte = pickle.load(f)

dic_prepro1 = {}
key_lst = []
val_lst = []
for i,j in lotte.items():
    key_lst.append(i)
    val_lst.append(j)
    

lotte_dic = pd.DataFrame([key_lst, val_lst],index=None).T
np.sum(lotte_dic.iloc[:,1]=='')
df_err = lotte_dic[lotte_dic.iloc[:,1]=='']
kwd_dic = lotte.copy()

del kwd_dic['11m']

for i in list(df_err.iloc[:,0]):
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
        print('err_kwd추가완료')
    else :
        key = driver.find_element_by_partial_link_text(check_txt)
        driver.implicitly_wait(50)
        key.send_keys(Keys.ENTER)
        driver.implicitly_wait(50)
        time.sleep(10)
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
            print('err_kwd추가완료')
        else:
            kwd_cls = driver.find_element_by_class_name("wrap-location")
            id_list = kwd_cls.find_elements_by_xpath("//div[@id]")
            string = str()
            for k in range(len(id_list)):
                if('divCat' in id_list[k].get_attribute("id")):
                    string = string + id_list[k].text + str('///')
            kwd_dic[i]=string
            print('dic추가완료')

kwd_dic['수입캔맥주'] = '주류'
kwd_dic['와인'] = '주류'
kwd_dic['하이트제로'] = '주류'
kwd_dic['맥주제로'] = '주류'
kwd_dic['하이트 캔맥주'] = '주류'
kwd_dic['하이트'] = '주류'
kwd_dic['11m'] = '가구 수납 조명 보수/조명 전구/컴팩트 전구'

kwd_dic['화이트제로']
with open(r'C:\Users\YongTaek\Desktop\real_dic1.pickle', 'wb') as f:
    pickle.dump(kwd_dic, f, pickle.HIGHEST_PROTOCOL)

df_err
with open(r'C:\Users\YongTaek\Desktop\err_kwd.pickle', 'rb') as f:
    lotte_err = pickle.load(f)

lotte_err.append(df_err.iloc[0,0])
lotte_err.append(df_err.iloc[1,0])
lotte_err.append(df_err.iloc[2,0])

del kwd_dic[df_err.iloc[0,0]]
del kwd_dic[df_err.iloc[1,0]]
del kwd_dic[df_err.iloc[2,0]]
del kwd_dic['11m']
lotte_err.append('생수\\')

len(kwd_dic)
with open('err_kwd.pickle', 'wb') as f:
    pickle.dump(lotte_err, f)