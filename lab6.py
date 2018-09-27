import csv
from sklearn import svm


reader = csv.reader(open('_svm-data.csv'),delimiter=',')
data_train = []
y_train = []
for row in reader:
    data_train.append([float(row[1]), float(row[2])])
    y_train.append(float(row[0]))

clf = svm.SVC(kernel='linear', C=100000, random_state=241)
clf.fit(data_train, y_train)
f = open('output/6answer.txt', 'w')
f.write("{0} {1} {2}".format(clf.support_[0]+1, clf.support_[1]+1, clf.support_[2]+1))

