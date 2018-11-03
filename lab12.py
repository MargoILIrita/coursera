import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from numpy import mean
from sklearn.metrics import r2_score

data = pandas.read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
y = data['Rings']
X = data.drop(columns=['Rings'])
kfold = KFold(n_splits=5 ,random_state=1, shuffle=True)
marks = []
find = 0
print('start for')
for i in range(1, 50, 1):
    regrs =  RandomForestRegressor(random_state=1, n_estimators=i, n_jobs=-1)
    temp = []
    for train_index, test_index in kfold.split(X,y):
        train_X = X.get_values()[train_index]
        test_X = X.get_values()[test_index]
        train_Y = y.get_values()[train_index]
        test_Y = y.get_values()[test_index]
        regrs.fit(train_X,train_Y)
        temp.append(r2_score(test_Y,regrs.predict(test_X)))
    marks.append(mean(temp))
    if 0.52 < mean(temp):
        find = i
        break
    print('finish ' + i.__str__())
print(marks)
print(find)

f = open("output/12answer-amount.txt", 'w')
f.write(find.__str__())