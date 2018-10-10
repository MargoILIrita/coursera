import csv
import sklearn.metrics as mt

reader = csv.reader(open('classification.csv'),delimiter=',')
true = []
pred = []
tp = 0
tn = 0
fp = 0
fn = 0
for row in reader:
    true.append(row[0])
    pred.append(row[1])
    if row[0] == 'true':
        continue
    if row[0] == '1':
        if row[1] == '1':
            tp+=1
        else:
            fn+=1
    else:
        if row[1] == '1':
            fp+=1
        else:
            tn+=1
true = list(map(int, true[1:]))
pred = list(map(int, pred[1:]))

aaaa = mt.confusion_matrix(true,pred)
print(aaaa)
print("tp {0} fp {1} fn {2} tn {3}".format(tp,fp,fn,tn))
f = open("output/9answer_metrics.txt", 'w')
f.write("{0} {1} {2} {3}".format(tp,fp,fn,tn))

# accuracy = round(mt.accuracy_score(true,pred),2)
# precision = round(mt.precision_score(true,pred),2)
# recall = round(mt.recall_score(true,pred),2)
# f1 = round(mt.f1_score(true,pred),2)
# f = open("output/9answer_mt.txt", 'w')
# f.write("{0} {1} {2} {3}".format(accuracy,precision,recall,f1))
#
# true = []
# logreg = []
# svm = []
# knn =[]
# tree = []
# reader = csv.reader(open('_scores.csv'),delimiter=',')
#
# feature_name = ''
# for row in reader:
#     if row[0] == 'true':
#         feature_name = dict.fromkeys(row[1:])
#         continue
#     true.append(row[0])
#     logreg.append(row[1])
#     svm.append(row[2])
#     knn.append(row[3])
#     tree.append(row[4])
# true = list(map(int, true))
# feature_name['score_logreg'] = list(map(float, logreg))
# feature_name['score_svm'] = list(map(float,svm))
# feature_name['score_knn'] = list(map(float, knn))
# feature_name['score_tree'] = list(map(float, tree))
#
# score = {mt.roc_auc_score(true,feature_name[x]):x for x in feature_name}
#
# f = open("output/9answer_max.txt", 'w')
# f.write(score[max(score.keys())])
#
# score = {x: mt.precision_recall_curve(true,feature_name[x]) for x in feature_name}
#
# def counting_score(metrics):
#     max = 0
#     for i in range(len(metrics[0])):
#         if metrics[1][i] >= 0.7 and metrics[0][i] > max:
#             max = metrics[0][i]
#     return max
#
# max_score = {counting_score(score[x]):x for x in score}
# f = open("output/9answer_70.txt", 'w')
# f.write(max_score[max(max_score.keys())])
