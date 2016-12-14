from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from sklearn.neural_network import MLPClassifier
import pickle

def neural_net(X, y):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    print (clf.fit(X, y))
    pickle.dump(clf, open('model/nn', 'wb'))

def main() :
    csv_file = 'data/dataset_final.csv'

    corpus = []
    author_vec = []

    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=";")

        for row in reader:
            corpus.append(row[1])
            author_vec.append(row[4])

    vectorizer = TfidfVectorizer(min_df=1)
    vec = vectorizer.fit_transform(corpus)

    neural_net(vec.toarray(), author_vec)

    #print (vec.toarray())
    #for i in vec.toarray():
    #    zeroes = True
    #    for j in i:
    #        if j != 0:
    #            zeroes = False
    #    if zeroes:
    #        print ("only zeroes")


    return

if __name__ == '__main__':
    main()