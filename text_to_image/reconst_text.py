import chainer
import numpy as np
traint = np.load('birds_txt.npy')
train_ind = [0]

xx = np.asarray(traint[train_ind])
o = xx.reshape((20,30))
x = chainer.Variable(xx)

import net
#model = netF.VAE(4000, 10, 2000, 800, 100)
model = net.VAE(600, 20, 300,100)
chainer.serializers.load_npz('birds_txt.npz', model)
mu, ln_var = model.encode(x)

y = model.decode(mu)

y = y.reshape((20,30))


#yy.shape



from gensim.models import word2vec
w2v = word2vec.Word2Vec.load('birds_txt.model')

"""
for w in y:
    reconst = w2v.similar_by_vector(w.data, topn=  1)
    print(reconst)
    print("================")
"""
reconst_words = [w2v.similar_by_vector(yy.data, topn=1)[0][0] for yy in y]
print(reconst_words)
txt2 = ' '.join(reconst_words)

print(txt2)

