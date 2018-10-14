import pandas
from numpy import corrcoef
from sklearn.decomposition import PCA

prices = pandas.read_csv('close_prices.csv',delimiter=',')
dj = pandas.read_csv('djia_index.csv', delimiter=',')
prices = prices.drop(columns=['date'])
print(prices.head())
print(prices.head())


pca = PCA(10)
pca.fit_transform(prices)

X = pca.transform(prices)
first = X[:,0]
f = open("output/11answer-first.txt", 'w')
f.write(round(corrcoef(first,dj['^DJI'])[0,1],2).__str__())

first_comp = pca.components_[0]
index = first_comp.tolist().index(first_comp.max())
print(prices.keys()[index])
f = open("output/11answer-name.txt", 'w')
f.write(prices.keys()[index])