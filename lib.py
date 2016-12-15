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

from math import sqrt

from stop_words import get_stop_words
from nltk.stem.snowball import EnglishStemmer

#
#
#
def clean_text(text):

    stemmer = EnglishStemmer()

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
    x = x.replace("'", '')
    x = x.replace('_', '')
    x = x.replace('?', '')
    x = x.replace('!', '')
    x = x.replace('(', '')
    x = x.replace(')', '')
    x = x.lower()
    x1 = ' '.join(stemmer.stem(w) for w in x.split(' '))
    return x1


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
            if row[0] != 'title':
                corpus.append(clean_text(row[1]))

                author = [0, 1]
                if int(row[4]) == 1:
                    author = [1, 0]

                author_vec.append(author)
                author_vec1.append(int(row[4]))

    vectorizer = CountVectorizer(min_df=25, stop_words = get_stop_words('en'))
    #vectorizer = TfidfVectorizer(stop_words = get_stop_words('en'))
    vec = vectorizer.fit_transform(corpus)

    print (vec.shape)
    #print (vectorizer.get_feature_names())

    return vectorizer, vec.toarray(), author_vec, author_vec1


def set_pybrain_nn(X, y):

    params_len = len(X[0])
    hidden_size = 150
    output_layer_num = 2
    epochs = 50

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

    trainer.trainUntilConvergence(verbose=True)
    trainer.train()

    #for i in range(epochs):
    #    mse = trainer.train()
    #    rmse = sqrt(mse)
    #    print("training RMSE, epoch {}: {}".format(i + 1, rmse))

    pickle.dump(net, open('model/nn_brain', 'wb'))

#
#
#
def set_neural_net(X, y):
    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
   # clf.fit(X, y)
    clf = MLPClassifier(hidden_layer_sizes=(100,2), random_state=1, warm_start=True)
    clf.fit(X, y)

    pickle.dump(clf, open('model/nn', 'wb'))

#
#
#
def set_svm(X, y):
    clf = svm.SVC(C=1, gamma=10, kernel='linear')
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
def get_pybrain_nn():
    return  pickle.load(open('model/nn_brain', 'rb'))

#
#
#
def get_svm():
    return  pickle.load(open('model/svm', 'rb'))