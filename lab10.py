import pandas
from sklearn.feature_extraction import DictVectorizer
import scipy.sparse
import sklearn.feature_extraction as fe
from sklearn.linear_model import Ridge


def updateStrings(name,dd):
    dd[name] = dd[name].str.lower()
    dd[name] = dd[name].replace('[^a-zA-Z0-9]', ' ', regex=True)

df = pandas.read_csv('salary-train.csv',delimiter=',')

updateStrings('FullDescription',df)
updateStrings('LocationNormalized',df)
updateStrings('ContractTime', df)
vectorized = fe.text.TfidfVectorizer()
vectorized.min_df = 5
X = vectorized.fit_transform(df['FullDescription'])
df['LocationNormalized'].fillna('nan', inplace=True)
df['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()
X_categ = enc.fit_transform(df[['LocationNormalized', 'ContractTime']].to_dict('records'))
train = scipy.sparse.hstack((X,X_categ))
print(train)

clf = Ridge(alpha=1, random_state=241)
clf.fit(train,df['SalaryNormalized'])

tdf = pandas.read_csv('salary-test-mini.csv',delimiter=',')
updateStrings('FullDescription',tdf)
updateStrings('LocationNormalized',tdf)
updateStrings('ContractTime', tdf)
X_test = vectorized.transform(tdf['FullDescription'])
tdf['LocationNormalized'].fillna('nan', inplace=True)
tdf['ContractTime'].fillna('nan', inplace=True)

X_categ_test = enc.transform(tdf[['LocationNormalized', 'ContractTime']].to_dict('records'))
test = scipy.sparse.hstack((X_test,X_categ_test))
f = open("output/10answer.txt", 'w')
res = clf.predict(test)
f.write("{0} {1}".format(round(res[0],2), round(res[1]),2))

