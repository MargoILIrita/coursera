from sklearn import neighbors
from sklearn import model_selection
from sklearn import preprocessing

f = open('wine.data')
wines = []
classes = []
for line in f:
    temp = line.split(',')
    classes.append(int(temp[0]))
    temp = temp[1:]
    elem = []
    for t in temp:
        elem.append(float(t))
    wines.append(elem)

generator = model_selection.KFold(shuffle=True, n_splits=5, random_state=42)
various = generator.split(wines,classes)
marks = []
for i in range(1,51,1):
    marks.append(model_selection.cross_val_score(
        neighbors.KNeighborsClassifier(n_neighbors=i), cv=generator, X=wines, y=classes).mean())
maxi = max(marks)
f = open('output/3knonnorm.txt', 'w')
f.write(str(marks.index(maxi)+1))
f = open('output/3valuenonnorm.txt', 'w')
f.write(round(maxi, 2).__str__())
wines = preprocessing.scale(wines)
marks = []
for i in range(1,51,1):
    marks.append(model_selection.cross_val_score(
        neighbors.KNeighborsClassifier(n_neighbors=i), cv=generator, X=wines, y=classes).mean())
maxi = max(marks)
f = open('output/3kwithnorm.txt', 'w')
f.write(str(marks.index(maxi)+1))
f = open('output/3valuewithnorm.txt', 'w')
f.write(round(maxi, 2).__str__())





