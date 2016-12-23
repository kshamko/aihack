import argparse
import os

import pickle as pickle
import sys

from functions import *

parser = argparse.ArgumentParser()
parser.add_argument('--model-folder', dest='model_folder', required=True)
parser.add_argument('--text', dest='text', required=True)


def predict(text, model_file, dict_file):
    net = pickle.load(open(model_file, 'rb'))
    wordDict = load_dict(dict_file)

    stemmed = stemm(text)

    x = []

    for word in wordDict:
        x.append(stemmed.count(word))

    pred = net.activate(x)

    res = max(enumerate(pred), key=lambda x: x[1])[0] + 1
    return res


if __name__ == '__main__':
    args = parser.parse_args()
    model_file = os.path.join(args.model_folder, 'model_nn.pkl')
    dict_file = os.path.join(args.model_folder, 'dict.txt')
    text = args.text
    print(predict(text, model_file, dict_file))
