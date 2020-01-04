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