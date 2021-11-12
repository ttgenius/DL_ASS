from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
# define training data
# sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
# 			['this', 'is', 'the', 'second', 'sentence'],
# 			['yet', 'another', 'sentence'],
# 			['one', 'more', 'sentence'],
# 			['and', 'the', 'final', 'sentence']]
sentences = [['Sam','went','to','the','store'],
             ['Then','Sam','came','to','the','theater'],
             ['I','went','to','the','theater']]
# train model
model = Word2Vec(sentences, min_count=1)
#print(model.wv['the'])
# fit a 2d PCA model to the vectors
#X = model[model.wv.vocab]
X= model.wv.vectors
print(X)
print(model.wv.index_to_key)
print(model.wv.key_to_index)
Y = [X[0],X[2],X[3],X[4],X[6]]
YS = [0,2,3,4,6]
print("Y",Y)
pca = PCA(n_components=2)
result = pca.fit_transform(X)
print("result",result)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = model.wv.index_to_key
# # words = [Sam, Came, Theater, Store, and Went]
# words = ['the','theater','went','Sam','came']
#words = ['the','theater','went','Sam','came']
print("words",words)
for i, word in enumerate(words):

    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()