import argparse
import os
import json

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import pickle as pickle
from math import sqrt

from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, RecurrentNetwork


parser = argparse.ArgumentParser()
parser.add_argument('--model-folder', dest='model_folder', required=True)

def _params_count(model_folder, file):
    count_params = 0
    with open(os.path.join(model_folder, file), 'r') as fp:
        for line in fp.readlines():
            count_params += 1

    fp.close()
    return count_params

def _build_params(filename, model_folder, batch_num, rows_per_step):
    print('build_params')
    count_params = _params_count(model_folder, 'dict.txt')
    print('start build')
    with open(filename, 'r') as fp:

        Xbig = []
        y = []

        i = 0
        for k,line in enumerate(fp.readlines()):

            if k >= rows_per_step * batch_num and k<rows_per_step * batch_num+rows_per_step:
                print(k)
                x = [0] * count_params
                line = json.loads(line)
                j = 1
                #line_len = len(line)

                for k,p in enumerate(line) :
                    #print(k)
                    #print(p)

                    if p == 'res':
                        y.append(k)
                    else:
                        x[int(p)] = k
                    #if j < line_len:
                    #    x.append(int(p))
                    #else:
                    #    y.append(int(p.strip('\n')))
                    j += 1

                Xbig.append(x)
                i += 1

    fp.close()

    return {'x': Xbig, 'y': y}

def _init_net(params_len, output_layer_num, hidden_size):
    # init and train
    net = FeedForwardNetwork()

    """ Next, we're constructing the input, hidden and output layers. """
    inLayer = LinearLayer(params_len)
    hiddenLayer = SigmoidLayer(hidden_size)
    hiddenLayer1 = SigmoidLayer(hidden_size)
    outLayer = LinearLayer(output_layer_num)

    """ (Note that we could also have used a hidden layer of type TanhLayer, LinearLayer, etc.)
    Let's add them to the network: """
    net.addInputModule(inLayer)
    net.addModule(hiddenLayer)
    net.addModule(hiddenLayer1)
    net.addOutputModule(outLayer)

    """ We still need to explicitly determine how they should be connected. For this we use the most
    common connection type, which produces a full connectivity between two layers (or Modules, in general):
    the 'FullConnection'. """

    in2hidden = FullConnection(inLayer, hiddenLayer)
    hidden2hidden = FullConnection(hiddenLayer, hiddenLayer1)
    hidden2out = FullConnection(hiddenLayer1, outLayer)

    net.addConnection(in2hidden)
    net.addConnection(hidden2hidden)
    net.addConnection(hidden2out)

    """ All the elements are in place now, so we can do the final step that makes our MLP usable,
    which is to call the 'sortModules()' method. """

    net.sortModules()

    # net = buildNetwork( params_len, hidden_size, 601, bias = True )
    return net


def main():
    args = parser.parse_args()

    hidden_size = 50
    epochs = 5

    dataset_len = _params_count(args.model_folder, 'params_train.txt')
    rows_per_step = 10000
    total_batches = dataset_len // rows_per_step


    params_len = _params_count(args.model_folder, 'dict.txt')
    output_layer_num = 601
    net = _init_net(params_len, output_layer_num, hidden_size)


    for batch_num in range(total_batches - 1):
        trainParams = _build_params(os.path.join(args.model_folder, 'params_train.txt'), args.model_folder, batch_num, rows_per_step)

        print('params ready')

        y = []

        for y_val in trainParams['y']:
            y_vec = [0] * output_layer_num
            y_vec[y_val - 1] = 1
            y.append(y_vec)

        print(len(trainParams['x']))
        print(len(y))

        # TODO: fix the number of pictures
        ds = SupervisedDataSet(params_len, output_layer_num)
        ds.setField('input', trainParams['x'])
        ds.setField('target', y)


        trainer = BackpropTrainer(net, ds)


        print("training for {} epochs...".format(epochs))

        #trainer.trainUntilConvergence(verbose=True)

        for i in range(epochs):
            mse = trainer.train()
            rmse = sqrt(mse)
            print("training RMSE, epoch {}: {}".format(i + 1, rmse))

    pickle.dump(net, open(os.path.join(args.model_folder, 'model_nn.pkl'), 'wb'))


if __name__ == '__main__':
    main()
