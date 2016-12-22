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
from random import shuffle
import matplotlib.pyplot as plt



from stop_words import get_stop_words
from nltk.stem.snowball import EnglishStemmer
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve

from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 8)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    print(test_scores)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

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
def set_svm(X, y, Xv, yv):

    X = preprocessing.scale(X)

    #C = [1, 5, 15, 20, 30, 100, 1000]
    #gamma = [0.001, 0.01, 0.1, 1]
    C = [5]
    gamma = [0.001]
    for c in C:
    	for g in gamma:
            print('\nC: %f, gamma: %f' % (c, g))
            clf = svm.SVC(C=c, gamma=g)#, kernel='linear')
            clf.fit(X, y)

            title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
            # SVC is more expensive so we do a lower number of CV iterations:
            cv = ShuffleSplit(n_splits=10, test_size=0.05, random_state=0)
            plt = plot_learning_curve(clf, title, X, y, (0.3, 1.01), cv=cv)

            print (clf.score(Xv, yv))
            plt.show()
            #pickle.dump(clf, open('model/svm_'+str(c)+'_'+str(g), 'wb'))


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
        reader = list(csv.reader(csvfile, delimiter=";"))

        shuffle(reader)
        #shuffle(reader)
        #shuffle(reader)

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


###########################################################
###########################################################
###########################################################

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