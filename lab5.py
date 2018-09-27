import  csv
from sklearn import linear_model as lm
from sklearn import metrics as metr
from sklearn import preprocessing as pr

reader = csv.reader(open('_perceptron-train.csv'),delimiter=',')
data_train = []
y_train = []
for row in reader:
    data_train.append([float(row[1]), float(row[2])])
    y_train.append(float(row[0]))

data_test = []
y_test = []
reader = csv.reader(open('_perceptron-test.csv'),delimiter=',')
for row in reader:
    data_test.append([float(row[1]), float(row[2])])
    y_test.append(float(row[0]))

clf = lm.Perceptron(random_state=241)
clf.fit(X=data_train, y=y_train)
non_norm_score = metr.accuracy_score(y_test, clf.predict(data_test))
print(non_norm_score)

scaler = pr.StandardScaler()
clf.fit(scaler.fit_transform(data_train),y_train)
norm_score = metr.accuracy_score(y_test, clf.predict(scaler.transform(data_test)))
print(norm_score)
f = open("output/5answer.txt", 'w')
f.write(round(norm_score - non_norm_score, 3).__str__())