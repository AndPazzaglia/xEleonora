#%%
from gensim.models import KeyedVectors
word_vectors = KeyedVectors.load("W2V.kv", mmap='r+')
#%%
vocabs = word_vectors.index_to_key
vectors = word_vectors.vectors

print(vocabs[0:30])

prova = word_vectors.get_index('di')
print(prova)

#%%
import matplotlib.pyplot as plt
import numpy as np

vocabs_index = [3, 4, 1000]
plt.figure()
for i in vocabs_index:
    vct = [v if np.abs(v) > 0.2 else 0 for v in vectors[i, :]]
    plt.plot(np.arange(300), vct, label=vocabs[i])
plt.legend(loc="upper left")
plt.show()
