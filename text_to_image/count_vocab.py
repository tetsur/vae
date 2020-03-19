from gensim.models import Word2Vec

model = Word2Vec.load("birds_txt.model")
v = model.wv.vocab
print(v)
print(len(v))
