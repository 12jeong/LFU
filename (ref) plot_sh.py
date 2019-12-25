# Summary Statistics
temp = raw_trans.copy()
temp['freq']=1
freq_PD1 = temp.groupby('clac_nm1').freq.sum()

idx = np.argsort(freq_PD1.values)[::-1][:ntype]
label = freq_PD1.index[idx]

plt.rcParams['figure.figsize'] = [20, 15]

# Horizontal Bar Chart
plt.barh(label, freq_PD1.values[idx])
plt.title("Frequncies of Product type1 in Data of Transaction",fontsize= 16)
plt.ylabel('Type of PD1', fontsize=15)
plt.yticks(label, label, fontsize=13, rotation=0)
plt.show()