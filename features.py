import lib

def main() :
    vectorizer, X, y, y1 = lib.get_training_set()
    Xv, yv, yv1, title = lib.get_test_set()

    #print("Train default NN")
    #lib.set_neural_net(X, y1, Xv, yv1)

    print("Train default SVM")
    lib.set_svm(X, y1, Xv, yv1)

    #print("Train pybrain NN")
    #lib.set_pybrain_nn(X, y)
    return

if __name__ == '__main__':
    main()