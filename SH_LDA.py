#  https://pypi.org/project/lda/
import numpy as np
import lda

docnames =  ['doc1', 'doc4', 'doc3', 'doc2']
vocab = ['science', 'mining', 'c', 'text', 'nlp', 'structures',
   'processing', 'matrix', 'r', 'algorithms', 'data',
   'programming', 'python', 'cleaning']
dtm = np.array([[0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 2, 0, 1, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0],
                [1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 1]])
  
model = lda.LDA(n_topics=3, n_iter=1000, random_state=1)

model.fit(dtm)

topic_word = model.topic_word_
n_top_words = 3

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    
    
    
    
    
#%% # libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
 
# I build a data set: 10 individuals and 5 variables for each
ind1=[5,10,3,4,8,10,12,1,9,4]
ind5=[1,1,13,4,18,5,2,11,3,8]
df = pd.DataFrame({ 'A':ind1, 'B':ind1 + np.random.randint(10, size=(10)) , 'C':ind1 + np.random.randint(10, size=(10)) , 'D':ind1 + np.random.randint(5, size=(10)) , 'E':ind1 + np.random.randint(5, size=(10)), 'F':ind5, 'G':ind5 + np.random.randint(5, size=(10)) , 'H':ind5 + np.random.randint(5, size=(10)), 'I':ind5 + np.random.randint(5, size=(10)), 'J':ind5 + np.random.randint(5, size=(10))})
df
# Calculate the correlation between individuals. We have to transpose first, because the corr function calculate the pairwise correlations between columns.
df_mart.columns.name=None
corr = df_mart.corr()
corr
 
# Transform it in a links data frame (3 columns only):
links = corr.stack().reset_index()
links.columns = ['var1', 'var2','value']
links
 
# Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)
links_filtered=links.loc[ (links['value'] > 0.8) & (links['var1'] != links['var2']) ]
links_filtered
 
# Build your graph
G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')
 
# Plot the network:
nx.draw(G, with_labels=True, node_color='orange', node_size=400, edge_color='black', linewidths=1, font_size=15)
    