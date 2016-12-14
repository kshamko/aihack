from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import csv

def file_get_contents(filename, use_include_path = 0, context = None, offset = -1, maxlen = -1):
    fp = open(filename,'rb')
    try:
        if (offset > 0):
            fp.seek(offset)
        ret = fp.read(maxlen)
        return ret
    finally:
        fp.close( )

def clean_text(text):

    x = text.replace(b'\n', b'')
    x = x.replace(b'\t', b'')
    x = x.replace(b'\r', b'')
    x = x.replace(b',', b'')
    x = x.replace(b'.', b'')
    x = x.replace(b'-', b'')
    x = x.replace(b',', b'')
    x = x.replace(b'"', b'')
    x = x.replace(b'_', b'')
    x = x.replace(b'?', b'')
    x = x.replace(b'!', b'')
    return x.lower()


def main() :

    text = file_get_contents('test/walrus.txt')
    text = clean_text(text)

    clf = pickle.load(open('model/nn', 'rb'))

    corpus = []
    csv_file = 'data/dataset_final.csv'
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=";")

        for row in reader:
            corpus.append(row[1])

    vectorizer = TfidfVectorizer(min_df=1)
    vectorizer.fit_transform(corpus)


    vec = vectorizer.transform([text])

    print (vec.toarray())

    print(clf.predict(vec.toarray()[0].reshape(1, -1)))

    return


if __name__ == '__main__':
    main()