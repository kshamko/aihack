import lib
import csv

def main() :
    #text = lib.file_get_contents('test/walrus.txt')
    text = []
    csv_file = 'data/dataset_test.csv'
    author_vec = []
    author_vec1 = []
    title = []

    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        for row in reader:
            text.append(lib.clean_text(row[1]))
            author = [0, 1]
            if int(row[2]) == 1:
                author = [1, 0]

            author_vec.append(author)
            author_vec1.append(int(row[2]))
            title.append(row[0])


    vectorizer, X, y, y1 = lib.get_training_set()
    vec = vectorizer.transform(text)

    clf = lib.get_neural_net()
    print(clf.score(vec.toarray(), author_vec1))

    print("SVM Predictions")
    clf1 = lib.get_svm()
    print(clf1.score(vec.toarray(), author_vec1))
    i = 0
    for x in vec.toarray():
        pred = clf1.predict([x])
        print(pred, author_vec1[i], title[i])
        i += 1


    print("PyBrain Predictions")
    clf2 = lib.get_pybrain_nn()
    i = 0;
    for x in vec.toarray():
        pred = clf2.activate(x)
        print(pred, int(not bool(max(enumerate(pred), key=lambda x: x[1])[0])), author_vec1[i], title[i])
        i += 1

    return


if __name__ == '__main__':
    main()