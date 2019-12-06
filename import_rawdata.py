#%%_import
%reset_-f
import_os
os.getcwd()
from_os_import_chdir

#_import_sys
import_pandas_as_pd
import_seaborn_as_sns_#_시각화
import_matplotlib.pyplot_as_plt
#_그래프의_스타일을_지정
plt.style.use('ggplot')
import_numpy_as_np
import_scipy_as_sp
import_sklearn
import_matplotlib_as_mpl
mpl.rcParams.update({'font.size':14})
plt.rc('font',family='Malgun_Gothic')_#_windows
%matplotlib_inline

pd.set_option('display.expand_frame_repr',_False)_#_expand_output_display_pd.df
#%%
data1_=_pd.read_csv("C:\\Users\\MYCOM\\Dropbox\\LFU\\제6회_L.POINT_Big_Data_Competition-분석용데이터-01.온라인_행동_정보.csv",low_memory=False)_
data2_=_pd.read_csv("C:\\Users\\MYCOM\\Dropbox\\LFU\\제6회_L.POINT_Big_Data_Competition-분석용데이터-02.거래_정보.csv")_
data3_=_pd.read_csv("C:\\Users\\MYCOM\\Dropbox\\LFU\\제6회_L.POINT_Big_Data_Competition-분석용데이터-03.고객_Demographic_정보.csv")_
data4_=_pd.read_csv("C:\\Users\\MYCOM\\Dropbox\\LFU\\제6회_L.POINT_Big_Data_Competition-분석용데이터-04.상품분류_정보.csv")_



