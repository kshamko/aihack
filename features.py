import lib

def main() :
    vectorizer, X, y, y1 = lib.get_training_set()
    lib.set_neural_net(X, y1)
    lib.set_svm(X, y1)
    return

if __name__ == '__main__':
    main()