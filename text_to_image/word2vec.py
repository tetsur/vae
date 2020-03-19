from gensim.models import word2vec
import os

sentence = word2vec.Text8Corpus('birds.txt')
model = word2vec.Word2Vec(sentence,  sg=1, size=30, min_count=1, window=5, hs=0, negative=15, iter=15)

model.save('birds_txt.model')
