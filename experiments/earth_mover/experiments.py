import sys
import os
import architectures
import architectures.simple
import datasets
import datasets.mnist
import iterators
from base import NeuralNet

def mnist_baseline():
    nn = NeuralNet(architectures.simple.light_net, num_classes=10, learning_rate=0.01, momentum=0.9, mode="x_ent", args={})
    nn.train(datasets.mnist.load_mnist("../../data/mnist.pkl.gz"), iterators.iterate, batch_size=32, num_epochs=10, out_dir="output/mnist_baseline", debug=True)

if __name__ == '__main__':

    #locals()[ sys.argv[1] ]

    mnist_baseline()