import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.init import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
from lasagne.updates import *
from lasagne.regularization import *
import helpers
from time import time
import iterators
import os
import layers

def dummy_net_fn(args):
    layer = InputLayer((None, 1, 28, 28))
    layer = DenseLayer(layer, num_units=5)
    return layer

class NeuralNet():
    def __init__(self, net_fn, num_classes, learning_rate, momentum, mode, args):
        assert mode in ["x_ent", "earth_mover"]
        self.num_classes = num_classes
        # network inputs/outputs
        self.l_out = net_fn(args)
        self.l_out_cum = layers.UpperRightOnesLayer(self.l_out)
        self.l_in = get_all_layers(self.l_out)[0]
        # theano variables
        X = T.tensor4('X')
        y = T.fmatrix('y')
        y_ord = T.fmatrix('y_ord')
        y_cum = T.fmatrix('y_cum')
        # ---
        self.params = get_all_params(self.l_out, trainable=True)
        self.net_out, self.net_out_cum = get_output([self.l_out, self.l_out_cum], {self.l_in: X})
        self.net_out_det, self.net_out_cum_det = get_output([self.l_out, self.l_out_cum], {self.l_in: X}, deterministic=True)
        if mode == "x_ent":
            train_loss = categorical_crossentropy(self.net_out, y).mean()
        elif mode == "earth_mover":
            train_loss = squared_error(self.net_out_cum, y_cum).mean()
        valid_loss_xent = categorical_crossentropy(self.net_out_det, y).mean()
        valid_loss_emd = categorical_crossentropy(self.net_out_cum_det, y_cum).mean()
        grads = T.grad(train_loss, self.params)
        self.learning_rate = theano.shared(floatX(learning_rate))
        self.momentum = momentum
        updates = nesterov_momentum(grads, self.params, learning_rate=self.learning_rate, momentum=self.momentum)
        train_fn = theano.function(inputs=[X, y, y_ord, y_cum], outputs=train_loss, updates=updates,
                                   on_unused_input='warn')
        xent_fn = theano.function(inputs=[X, y], outputs=valid_loss_xent)
        emd_fn = theano.function(inputs=[X, y_cum], outputs=valid_loss_emd)
        dist_fn = theano.function(inputs=[X], outputs=self.net_out_det)
        self.fns = {
            "train_fn": train_fn,
            "xent_fn": xent_fn,
            "emd_fn": emd_fn,
            "dist_fn": dist_fn
        }

    def load_weights_from(self, filename):
        with open(filename) as g:
            set_all_param_values(self.l_out, pickle.load(g))

    def _create_results_file(self, out_dir, filename):
        out_file = "%s/%s.txt" % (out_dir, filename)
        if os.path.exists(out_file):
            f_out_file = open(out_file, "a")
        else:
            f_out_file = open(out_file, "wb")
        return f_out_file

    def train(self, data, iterator_fn, batch_size, num_epochs, out_dir, schedule={}, debug=False):
        header = ["train_loss", "valid_xent", "valid_emd", "valid_accuracy", "learning_rate", "time"]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        f_out_file = self._create_results_file(out_dir, "results")
        f_out_file.write(",".join(header) + "\n")
        print ",".join(header)
        Xt, yt, Xv, yv = data
        train_fn, xent_fn, emd_fn, dist_fn = self.fns["train_fn"], self.fns["xent_fn"], self.fns["emd_fn"], self.fns["dist_fn"]
        for epoch in range(num_epochs):
            t0 = time()
            if epoch+1 in schedule:
                self.learning_rate.set_value( floatX(schedule[epoch+1]) )
            train_losses = []
            for Xb, yb in iterator_fn(Xt, yt, bs=batch_size, num_classes=self.num_classes):
                yb_ord, yb_cum = helpers.one_hot_to_ord(yb), helpers.one_hot_to_cmf(yb)
                train_losses.append(train_fn(Xb, yb, yb_ord, yb_cum))
                if debug:
                    break
            valid_xent_losses = []
            valid_emd_losses = []
            valid_correct = []
            for Xb, yb in iterator_fn(Xv, yv, bs=batch_size, num_classes=self.num_classes):
                yb_ord, yb_cum = helpers.one_hot_to_ord(yb), helpers.one_hot_to_cmf(yb)
                valid_xent_losses.append( xent_fn(Xb, yb) )
                valid_emd_losses.append( emd_fn(Xb, yb_cum) )
                valid_correct += ( np.argmax(dist_fn(Xb),axis=1) == np.argmax(yb,axis=1) ).tolist()
                #print valid_correct
                if debug:
                    break
            to_write = "%f,%f,%f,%f,%f,%f" % \
                       (np.mean(train_losses),
                        np.mean(valid_xent_losses),
                        np.mean(valid_emd_losses),
                        np.mean(valid_correct),
                        self.learning_rate.get_value(),
                        time()-t0)
            f_out_file.write("%s\n" % to_write)
            print to_write

    def dump_dists(self, X, iterator_fn, batch_size, out_file):
        """
        :param X: input data
        :param out_file: p(y|x)
        :return:
        """
        with open(out_file) as f:
            for Xb, _ in iterator_fn(X, X, bs=batch_size, num_classes=self.num_classes):
                dists = self.fns["dist_fn"](Xb)
                for row in dists:
                    row = [ str(elem) for elem in row.tolist() ]
                    f.write(",".join(row) + "\n")

if __name__ == '__main__':
    import datasets.mnist
    import architectures.simple

    print architectures.simple.light_net

    #data, iterator_fn, batch_size, num_epochs, out_file,

    nn = NeuralNet(architectures.simple.light_net, num_classes=10, learning_rate=0.01, momentum=0.9, mode="earth_mover", args={})
    print nn

    nn.train(
        datasets.mnist.load_mnist("../../data/mnist.pkl.gz"),
        iterators.iterate, batch_size=32, num_epochs=10, out_dir="/tmp", debug=False)



    #xin = np.ones((10, 1, 28, 28))
    #nn.dump_dists(X=xin, iterator_fn=iterators.iterate, batch_size=5, out_file="/tmp/test.txt")