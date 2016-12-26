import lib
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

def main() :

    X, y, y1, title = lib.get_test_set()

    print("SVM Predictions")
    clf = lib.get_svm()
    print(clf.score(X, y1))
    i = 0
    p = []
    for x in X:
        [pred] = clf.predict([x])
        #proba = clf.classes_, clf.predict_proba([x])#clf.predict_proba([x])
        [des] = clf.decision_function([x])
        p.append(pred)



        print(pred, y1[i], des, (50 + abs(des)*100), title[i])
        i += 1


    #print("F1: ", f1_score(y1, p))
    #print("Recall: ", recall_score(y1, p))
    #print("Precision: ", precision_score(y1, p, labels=['p','j']))
    return


if __name__ == '__main__':
    main()