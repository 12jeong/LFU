from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

# 롯데마트 들어가기
driver = webdriver.Chrome(r'C:\Users\YongTaek\Desktop\chromedriver_win32\chromedriver')
driver.implicitly_wait(2)
driver.get('http://www.lottemart.com')

# 레쓰비 검색
div_elem = driver.find_element_by_class_name('swindow')
elem = div_elem.find_element_by_id("searchTerm")
elem.send_keys("anf 연어감자")

# 검색 버튼 클릭
div_click = driver.find_element_by_class_name('swindow')
div_click.find_element(By.XPATH, '//button').click()

# 첫 번째 상품 진입
item = driver.find_element_by_class_name("prod-name")
button = item.find_element_by_css_selector('a')
button.click()

# a tag 텍스트 긁어오기
driver.find_element_by_id("selCat1").text
driver.find_element_by_id("selCat2").text
driver.find_element_by_id("selCat3").text

''' 타고 들어가는거
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')
notices = soup.select('div.p_inr > div.p_info > a ')
notices '''
# 엔터 : button.send_keys('\n')

