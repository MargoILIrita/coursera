import numpy
from sklearn import datasets
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import neighbors

boston =  datasets.load_boston()
features = preprocessing.scale(boston.data)
generator = model_selection.KFold(shuffle=True, n_splits=5, random_state=42)
marks = {}
classifier = neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski')
for p in numpy.linspace(1,10,200):
    classifier.p = p
    marks[model_selection.cross_val_score( classifier,
        cv=generator, X=features, y=boston.target, scoring='neg_mean_squared_error').max()] = p
for key in marks:
    print("{0} {1}".format(key, marks[key]))
f = open('output/4answer.txt', 'w')
f.write(str(round(marks[max(marks.keys())],2)))