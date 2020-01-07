
import pickle
import pandas as pd
import numpy as np

with open(r'C:\Users\YongTaek\Desktop\fin_dic.pickle', 'rb') as f:
    fin_dic = pickle.load(f)

with open(r'C:\Users\YongTaek\Desktop\real_dic2.pickle', 'rb') as f:
    real = pickle.load(f)

name = []
goods = []
for i, j in fin_dic.items():
    name.append(i)
    goods.append(j)

# 상품 소분류 카테고리 축소
df_goods = pd.DataFrame([name, goods], index=None).T
df_goods.columns = ['name', 'goods']
df_goods


df_goods['first_cls'] = df_goods.goods
df_goods.first_cls.value_counts()[df_goods.first_cls.value_counts()==1].head(60)
df_goods.first_cls.value_counts()[df_goods.first_cls.value_counts()==1].tail(40)
# 컵
df_goods.first_cls[df_goods.first_cls == 'Tableware water bottle']='Supplies Cup'
df_goods.first_cls[df_goods.first_cls == 'Shot glass']='Supplies Cup'
df_goods.first_cls[df_goods.first_cls == 'cup']='Supplies Cup'
df_goods.first_cls[df_goods.first_cls == 'Thermos']='Supplies Cup' # 보온병
df_goods.first_cls[df_goods.first_cls == 'Glass Rock']='Supplies Cup'

# 이름만 살짝 수정
df_goods.first_cls[df_goods.first_cls == 'Diaper-Lottette']='Diaper'
df_goods.first_cls[df_goods.first_cls == 'Mushroom set']='Mushroom'


# 공통 패션의류
df_goods.first_cls[df_goods.first_cls == 'Fashion miscellaneous goods']='Fashion&dress'
df_goods.first_cls[df_goods.first_cls == 'One piece suit']='Fashion&dress'
df_goods.first_cls[df_goods.first_cls == 'Scarf scarf neck warmer']='Fashion&dress'
df_goods.first_cls[df_goods.first_cls == 'Pants Banding Pants']='Fashion&dress'
df_goods.first_cls[df_goods.first_cls == 'Trouser pants']='Fashion&dress'
df_goods.first_cls[df_goods.first_cls == 'Southern Junior Underwear']='Fashion&dress'
df_goods.first_cls[df_goods.first_cls == 'Jogori Bodysuit Space Suit']='Fashion&dress'
df_goods.first_cls[df_goods.first_cls == 'Bag miscellaneous goods']='Fashion&dress'
df_goods.first_cls[df_goods.first_cls == 'Casual belt']='Fashion&dress'
df_goods.first_cls[df_goods.first_cls == 'Cardigan']='Fashion&dress'

# 애완용품
df_goods.first_cls[df_goods.first_cls == 'Cat Functional Feeds']='Pet'
df_goods.first_cls[df_goods.first_cls == 'Pet food']='Pet'
df_goods.first_cls[df_goods.first_cls == 'Cat Functional Feeds']='Pet'
df_goods.first_cls[df_goods.first_cls == 'Cat Functional Feeds']='Pet'
df_goods.first_cls[df_goods.first_cls == 'Dog Pad']='Pet'
df_goods.first_cls[df_goods.first_cls == 'Dog Toys']='Pet'



# 여자 패션의류
df_goods.first_cls[df_goods.goods == 'Tights leggings']='Fashion&dress_W'
df_goods.first_cls[df_goods.goods == 'Nowire Bra']='Fashion&dress_W'
df_goods.first_cls[df_goods.goods == 'Sports seamless bra']='Fashion&dress_W'
df_goods.first_cls[df_goods.goods == "Women's Bra Panties Set"]='Fashion&dress_W'

# 악세서리
df_goods.first_cls[df_goods.first_cls == "Children's Hair Accessories"]='Accesoories'
df_goods.first_cls[df_goods.first_cls == "Adult Hair Accessories"]='Accesoories'

# 남성 패션/의류
df_goods.first_cls[df_goods.first_cls == "Men's shirts"]='Fashion&dress_M'
df_goods.first_cls[df_goods.first_cls == "Mens home easywear"]='Fashion&dress_M'
df_goods.first_cls[df_goods.first_cls == "Mens jacket jumper"]='Fashion&dress_M'
df_goods.first_cls[df_goods.first_cls == "Men's trousers"]='Fashion&dress_M'
df_goods.first_cls[df_goods.first_cls == "Men's Jacket Jumpers"]='Fashion&dress_M'
df_goods.first_cls[df_goods.first_cls == "Men's Wallet Belt"]='Fashion&dress_M'
df_goods.first_cls[df_goods.first_cls == "Men's pants bottoms"]='Fashion&dress_M'

# 전자 장비
df_goods.first_cls[df_goods.first_cls == "Cell Phone Supplies"]='Electronic Equipment'
df_goods.first_cls[df_goods.first_cls == "headphone"]='Electronic Equipment'
df_goods.first_cls[df_goods.first_cls == "e-coupon"]='Electronic Equipment'
df_goods.first_cls[df_goods.first_cls == "General electric fan"]='Electronic Equipment'
df_goods.first_cls[df_goods.first_cls == "Microwave oven"]='Electronic Equipment'
df_goods.first_cls[df_goods.first_cls == "General Instant Camera Telescope"]='Electronic Equipment'
df_goods.first_cls[df_goods.first_cls == "Electric Pressure Cooker Electric Rice Cooker"]='Electronic Equipment'
df_goods.first_cls[df_goods.first_cls == "dish wash machine"]='Electronic Equipment'
df_goods.first_cls[df_goods.first_cls == "Fryer Airfryer"]='Electronic Equipment'
df_goods.first_cls[df_goods.first_cls == "Coffee Maker Bean Capsule Machine"]='Electronic Equipment'
df_goods.first_cls[df_goods.first_cls == "USB memory"]='Electronic Equipment'
df_goods.first_cls[df_goods.first_cls == "robotic vacuum"]='Electronic Equipment'
df_goods.first_cls[df_goods.first_cls == "External Hard SSD HDD"]='Electronic Equipment'
df_goods.first_cls[df_goods.first_cls == "printer"]='Electronic Equipment'
df_goods.first_cls[df_goods.first_cls == "Air Purifier Air Washer"]='Electronic Equipment'
df_goods.first_cls[df_goods.first_cls == "Seasoning machine"]='Electronic Equipment'
df_goods.first_cls[df_goods.first_cls == "dehumidifier"]='Electronic Equipment'
df_goods.first_cls[df_goods.first_cls == "Dyson"]='Electronic Equipment'
df_goods.first_cls[df_goods.first_cls == "Hot air heater"]='Electronic Equipment'
df_goods.first_cls[df_goods.first_cls == "Electric fan"]='Electronic Equipment'
df_goods.first_cls[df_goods.first_cls == "Cable accessories"]='Electronic Equipment'
df_goods.first_cls[df_goods.first_cls == "Rhythm Set Recorder"]='Electronic Equipment'

#신발 
df_goods.first_cls[df_goods.first_cls == "sandal"]='shoes'
df_goods.first_cls[df_goods.first_cls == "sports shoes"]='shoes'
df_goods.first_cls[df_goods.first_cls == "Flat shoes"]='shoes'
df_goods.first_cls[df_goods.first_cls == "casual shoes"]='shoes'
df_goods.first_cls[df_goods.first_cls == "Men's shoes"]='shoes'

# 스포츠 도구
df_goods.first_cls[df_goods.first_cls == "Golf Practice Goods"]='Sports Equipment'
df_goods.first_cls[df_goods.first_cls == "basketball"]='Sports Equipment'
df_goods.first_cls[df_goods.first_cls == "Baseball bat"]='Sports Equipment'
df_goods.first_cls[df_goods.first_cls == "Sports watch"]='Sports Equipment'
df_goods.first_cls[df_goods.first_cls == "Shoulder running"]='Sports Equipment'
df_goods.first_cls[df_goods.first_cls == "Baseball Others"]='Sports Equipment'
df_goods.first_cls[df_goods.first_cls == "Safety Goods Guard"]='Sports Equipment'
df_goods.first_cls[df_goods.first_cls == "bicycle"]='Sports Equipment'

# 화장품/뷰티
df_goods.first_cls[df_goods.first_cls == "Moisturizing Beauty Tissue"]='cosmetics'
df_goods.first_cls[df_goods.first_cls == "Makeup accessories"]='cosmetics'
df_goods.first_cls[df_goods.first_cls == "Moisturizing Beauty Tissue"]='cosmetics'
df_goods.first_cls[df_goods.first_cls == "Sheet mask"]='cosmetics'
df_goods.first_cls[df_goods.first_cls == "Sunspray"]='cosmetics'
df_goods.first_cls[df_goods.first_cls == "Makeup accesorries"]='cosmetics'
df_goods.first_cls[df_goods.first_cls == "Eye makeup"]='cosmetics'
df_goods.first_cls[df_goods.first_cls == "Body mist"]='cosmetics'
df_goods.first_cls[df_goods.first_cls == "Sunscreen sun lotion"]='cosmetics'
df_goods.first_cls[df_goods.first_cls == "Makeup accessories"]='cosmetics'
df_goods.first_cls[df_goods.first_cls == "Essence Ampoule Serum"]='cosmetics'
df_goods.first_cls[df_goods.first_cls == "Foot care"]='cosmetics'
df_goods.first_cls[df_goods.first_cls == "Gel"]='cosmetics'
df_goods.first_cls[df_goods.first_cls == "Gel wax"]='cosmetics'
df_goods.first_cls[df_goods.first_cls == "Hairbrush comb"]='cosmetics'
df_goods.first_cls[df_goods.first_cls == "Cleansing water oil"]='cosmetics'
df_goods.first_cls[df_goods.first_cls == "Cleansing accessory"]='cosmetics'
df_goods.first_cls[df_goods.first_cls == "Cleansing sun care"]='cosmetics'
df_goods.first_cls[df_goods.first_cls == "Whitening toothpaste"]='cosmetics'
df_goods.first_cls[df_goods.first_cls == "Hairbrush cap roll"]='cosmetics'

# 가정용 생필품 
df_goods.first_cls[df_goods.first_cls == "Fabric softener"]='Home Necessity'
df_goods.first_cls[df_goods.first_cls == "shampoo"]='Home Necessity'
df_goods.first_cls[df_goods.first_cls == "Pillow cover"]='Home Necessity'
df_goods.first_cls[df_goods.first_cls == "Bottle Brush Tongs Disinfection Supplies"]='Home Necessity'
df_goods.first_cls[df_goods.first_cls == "Bottle Brush Tongs Disinfection Supplies"]='Home Necessity'
df_goods.first_cls[df_goods.first_cls == "Lock & lock"]='Home Necessity'
df_goods.first_cls[df_goods.first_cls == "Interdental toothbrush"]='Home Necessity'
df_goods.first_cls[df_goods.first_cls == "House slippers"]='Home Necessity'
df_goods.first_cls[df_goods.first_cls == "Cookware"]='Home Necessity'
df_goods.first_cls[df_goods.first_cls == "Pot set"]='Home Necessity'
df_goods.first_cls[df_goods.first_cls == "Washboard clothespins"]='Home Necessity'
df_goods.first_cls[df_goods.first_cls == "General cleaner"]='Home Necessity'
df_goods.first_cls[df_goods.first_cls == "Toilet bowl cleaner"]='Home Necessity'
df_goods.first_cls[df_goods.first_cls == "Scaffolding"]='Home Necessity'

df_goods.first_cls[df_goods.first_cls == "Hi-Mart Cool Mat Pest Control Insect Killer"]='Insecticide'

# 양념
df_goods.first_cls[df_goods.first_cls == "Jam syrup"]='seasoning'
df_goods.first_cls[df_goods.first_cls == "Black pepper spice"]='seasoning'
df_goods.first_cls[df_goods.first_cls == "sandal"]='shoes'
df_goods.first_cls[df_goods.first_cls == "sandal"]='shoes'
df_goods.first_cls[df_goods.first_cls == "sandal"]='shoes'


# 캠핑
df_goods.first_cls[df_goods.first_cls == "blanket"]='Camping'
df_goods.first_cls[df_goods.first_cls == "Disposable Chopstick Cutlery"]='Camping'
df_goods.first_cls[df_goods.first_cls == "Basket"]='Camping'
df_goods.first_cls[df_goods.first_cls == "Charcoal BBQ Supplies"]='Camping'
df_goods.first_cls[df_goods.first_cls == "Camping Chair Outdoor Goods"]='Camping'

# 가구/인테리어
df_goods.first_cls[df_goods.first_cls == "Dresser dresser bedroom furniture"]='interior'
df_goods.first_cls[df_goods.first_cls == "curtain"]='interior'
df_goods.first_cls[df_goods.first_cls == "Curtain Accessories"]='interior'

df_goods.iloc[9315:9320,:]
df_goods[df_goods.goods=='Hi-Mart Cool Mat Pest Control Insect Killer']

df_goods.to_csv(r"C:\Users\YongTaek\Desktop\df_goods.csv")