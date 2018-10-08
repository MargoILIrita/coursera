import sklearn
from sklearn import datasets
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
import numpy as np

newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )
print("Downloaded")
vectorized = sklearn.feature_extraction.text.TfidfVectorizer()
X = vectorized.fit_transform(newsgroups.data,y=newsgroups.target)
# print("Transformed")
# grid = {'C': np.power(10.0, np.arange(-5, 6))}
# cv = KFold(n_splits=5, shuffle=True, random_state=241)
# clf = SVC(kernel='linear', random_state=241)
# gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
# print("Start fit")
# gs.fit(X, newsgroups.target)
# max = gs.grid_scores_[0]
# for a in gs.grid_scores_:
#     if a.mean_validation_score > max.mean_validation_score:
#         max = a
#     print("{0} {1}".format(a.mean_validation_score, a[0]['C']))
# print(max)
clf = SVC(kernel='linear', random_state=241, C=1)
clf.fit(X, newsgroups.target)
word_indexes = np.argsort(np.abs(np.asarray(clf.coef_.todense())).reshape(-1))[-10:]

words = [vectorized.get_feature_names()[i] for i in word_indexes]
words = np.sort(words)
strW = ''
for a in words:
    strW+= a.strip()+','

f = open("output/7answer.txt", 'w')
f.write(strW[:-1])




