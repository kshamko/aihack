from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import numpy as np

import csv
import pickle

#
#
#
def clean_text(text):
    x = text
    x = x.replace('\n', '')
    x = x.replace("||", ' ')
    x = x.replace('\t', '')
    x = x.replace('\r', '')
    x = x.replace(',', '')
    x = x.replace('.', '')
    x = x.replace('-', '')
    x = x.replace(',', '')
    x = x.replace('"', '')
    x = x.replace('_', '')
    x = x.replace('?', '')
    x = x.replace('!', '')
    return x.lower()

#
#
#
def file_get_contents(filename, use_include_path = 0, context = None, offset = -1, maxlen = -1):
    fp = open(filename,'rb')
    try:
        if (offset > 0):
            fp.seek(offset)
        ret = fp.read(maxlen)
        return ret
    finally:
        fp.close( )


#
#
#
def get_training_set() :

    csv_file = 'data/dataset_final.csv'
    corpus = []
    author_vec = []
    author_vec1 = []

    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=";")

        for row in reader:
            corpus.append(clean_text(row[1]))

            author = [0, 1]
            if row[4] == 1:
                author = [1, 0]

            author_vec.append(author)
            author_vec1.append(row[4])

    vectorizer = TfidfVectorizer(min_df=1)
    vec = vectorizer.fit_transform(corpus)

    return vectorizer, vec.toarray(), author_vec, author_vec1



#
#
#
def set_neural_net(X, y):
    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
   # clf.fit(X, y)
    clf = MLPClassifier(hidden_layer_sizes=(15,2), random_state=1, warm_start=True)
    for i in range(100):
        clf.fit(X, y)

    pickle.dump(clf, open('model/nn', 'wb'))

#
#
#
def set_svm(X, y):
    clf = svm.SVC(C=100, gamma=0.01, kernel='linear')
    clf.fit(np.array(X), np.array(y))
    pickle.dump(clf, open('model/svm', 'wb'))

#
#
#
def get_neural_net():
    return  pickle.load(open('model/nn', 'rb'))

#
#
#
def get_svm():
    return  pickle.load(open('model/svm', 'rb'))