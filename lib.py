from sklearn.neural_network import MLPClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn import svm
import numpy as np

from pybrain.datasets import SupervisedDataSet, ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, RecurrentNetwork

import csv
import pickle
from sklearn import preprocessing
from math import sqrt

from stop_words import get_stop_words
from nltk.stem.snowball import EnglishStemmer
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import validation_curve

#
#
#
def clean_text(text, stemmer = None):

    #stemmer = EnglishStemmer()
    text = text.replace("||", ' ').lower()

    items_to_remove = ['\n', '\t', '\r', ',', '.', '-', ',',
        '"', "'ve", "'ll", '_', '?', '!', '(', ')', 'yoko',
    ]

    for item in items_to_remove:
        text = text.replace(item, '')


    if stemmer is not None:
        text = ' '.join(stemmer.stem(w) for w in text.split(' '))

    return text


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
    corpus, author_vec, author_vec1, title = _iterate_dataset_file(csv_file)

    vectorizer = CountVectorizer()#TfidfVectorizer(min_df=1)
    vec = vectorizer.fit_transform(corpus)
    return vectorizer, vec.toarray(), author_vec, author_vec1

#
#
#
def get_test_set() :
    csv_file = 'data/dataset_test.csv'
    text, author_vec, author_vec1, title = _iterate_dataset_file(csv_file, 2)
    vectorizer, X, y, y1 = get_training_set()
    vec = vectorizer.transform(text)

    return vec.toarray(), author_vec, author_vec1, title

#
#
#
def set_pybrain_nn(X, y):

    params_len = len(X[0])

    print(params_len)
    hidden_size = 100
    output_layer_num = 2
    epochs = 200

    # init and train
    net = FeedForwardNetwork()

    """ Next, we're constructing the input, hidden and output layers. """
    inLayer = LinearLayer(params_len)
    hiddenLayer = SigmoidLayer(hidden_size)
    hiddenLayer1 = SigmoidLayer(hidden_size)
    hiddenLayer2 = SigmoidLayer(hidden_size)
    outLayer = LinearLayer(output_layer_num)


    """ (Note that we could also have used a hidden layer of type TanhLayer, LinearLayer, etc.)
    Let's add them to the network: """
    net.addInputModule(inLayer)
    net.addModule(hiddenLayer)
    net.addModule(hiddenLayer1)
    net.addModule(hiddenLayer2)
    net.addOutputModule(outLayer)

    """ We still need to explicitly determine how they should be connected. For this we use the most
    common connection type, which produces a full connectivity between two layers (or Modules, in general):
    the 'FullConnection'. """

    in2hidden = FullConnection(inLayer, hiddenLayer)
    hidden2hidden = FullConnection(hiddenLayer, hiddenLayer1)
    hidden2hidden1 = FullConnection(hiddenLayer1, hiddenLayer2)
    hidden2out = FullConnection(hiddenLayer2, outLayer)

    net.addConnection(in2hidden)
    net.addConnection(hidden2hidden)
    net.addConnection(hidden2hidden1)
    net.addConnection(hidden2out)

    """ All the elements are in place now, so we can do the final step that makes our MLP usable,
    which is to call the 'sortModules()' method. """

    net.sortModules()

    #ds = SupervisedDataSet(params_len, output_layer_num)
    ds = ClassificationDataSet(params_len, output_layer_num, nb_classes=2)
    ds.setField('input', X)
    ds.setField('target', y)

    trainer = BackpropTrainer(net, ds)

    print("training for {} epochs...".format(epochs))

    #trainer.trainUntilConvergence(verbose=True)
    #trainer.train()

    for i in range(epochs):
        mse = trainer.train()
        rmse = sqrt(mse)
        print("training RMSE, epoch {}: {}".format(i + 1, rmse))

    pickle.dump(net, open('model/nn_brain', 'wb'))

#
#
#
def set_neural_net(X, y, Xv, yv):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,100), random_state=1, warm_start=True)
    clf.fit(X, y)
    print(clf.score(Xv, yv))
    pickle.dump(clf, open('model/nn', 'wb'))

#
#
#
def set_svm(X, y, Xv, yv):

    X = preprocessing.scale(X)

    C = [1, 5, 15, 20, 30, 100, 1000]
    gamma = [0.001, 0.01, 0.1, 1]
    for c in C:
    	for g in gamma:
            print('\nC: %f, gamma: %f' % (c, g))
            clf = svm.SVC(C=c, gamma=g)#, kernel='linear')
            clf.fit(X, y)
            print (clf.score(Xv, yv))
            pickle.dump(clf, open('model/svm_'+str(c)+'_'+str(g), 'wb'))

#
#
#
def get_neural_net():
    return  pickle.load(open('model/nn', 'rb'))

#
#
#
def get_pybrain_nn():
    return  pickle.load(open('model/nn_brain', 'rb'))

#
#
#
def get_svm():
    return  pickle.load(open('model/svm_5_0.001', 'rb'))

#
#
#
def _iterate_dataset_file(file, author_index = 4):
    corpus = []
    corpus_j = []
    corpus_p = []
    author_vec = []
    author_vec1 = []
    title = []

    with open(file) as csvfile:
        reader = csv.reader(csvfile, delimiter=";")

        for row in reader:
            if row[0] != 'title':
                corpus.append(clean_text(row[1]))
                title.append(row[0])

                author = [0, 1]
                if int(row[author_index]) == 1:
                    corpus_j.append(clean_text(row[1]))
                    author = [1, 0]
                else:
                    corpus_p.append(clean_text(row[1]))

                author_vec.append(author)
                author_vec1.append(int(row[author_index]))

    return corpus, author_vec, author_vec1, title