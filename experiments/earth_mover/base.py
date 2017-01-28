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
import sys
import pdb
import pickle

def dummy_net_fn(args):
    layer = InputLayer((None, 1, 28, 28))
    layer = DenseLayer(layer, num_units=5)
    return layer

def test_iterator(data, iterator_fn, num_classes):
    Xt, yt, Xv, yv = data
    for Xb, yb in iterator_fn(Xt, yt, bs=batch_size, num_classes=num_classes):
        yield Xb, yb

class NeuralNet():

    def print_network(self, l_out):
        for layer in get_all_layers(l_out):
            print layer, layer.output_shape

    def _pmf_to_cmf(self, l_out):
        return layers.UpperRightOnesLayer(l_out)

    def _pmf_to_exp(self, l_out):
        l_exp = DenseLayer(l_out, num_units=1, nonlinearity=linear)
        mat = np.asarray([[i] for i in range(self.num_classes)]).astype("float32")
        l_exp.W.set_value(mat)
        return l_exp

    def _pmf_to_sq_err(self, l_out):
        def sigm_scaled_nonlinearity(x):
            return sigmoid(x)*(self.num_classes-1)
        l_exp = DenseLayer(l_out, num_units=1, nonlinearity=sigm_scaled_nonlinearity)
        pass

    def __init__(self, net_fn, num_classes, optimiser=nesterov_momentum,
                     optimiser_args={"learning_rate":theano.shared(floatX(0.01)),"momentum":0.9}, mode="x_ent", args={}, debug=False):
        assert mode in ["x_ent", "emd2", "emd22", "xemd2", "exp", "qwk"]
        self.num_classes = num_classes
        self.learning_rate = optimiser_args["learning_rate"]
        self.debug = debug
        # network inputs/outputs
        self.l_out = net_fn(args)
        if self.debug:
            self.print_network(self.l_out)
        self.l_out_cum = self._pmf_to_cmf(self.l_out)
        self.l_out_exp = self._pmf_to_exp(self.l_out)
        self.l_in = get_all_layers(self.l_out)[0]
        self.l_out_endpt = None
        # theano variables
        X = T.tensor4('X')
        y = T.fmatrix('y')
        y_ord = T.fmatrix('y_ord')
        y_cum = T.fmatrix('y_cum')
        y_int = T.ivector('y_int')
        # ---
        self.params = get_all_params(self.l_out, trainable=True)
        if self.debug:
            print "params: ", self.params
        # TODO: put these in a dict??
        self.net_out, self.net_out_cum, self.net_out_exp = \
            get_output([self.l_out, self.l_out_cum, self.l_out_exp], {self.l_in: X})
        self.net_out_det, self.net_out_cum_det, self.net_out_exp_det = \
            get_output([self.l_out, self.l_out_cum, self.l_out_exp], {self.l_in: X}, deterministic=True)
        if mode == "x_ent":
            if self.debug:
                print "train_loss: x_ent"
            train_loss = categorical_crossentropy(self.net_out, y).mean()
            self.l_out_endpt = self.l_out
        elif mode == "emd2":
            if self.debug:
                print "train_loss: emd2"
            #train_loss = squared_error(self.net_out_cum, y_cum).mean()
            train_loss = squared_error(self.net_out_cum, y_cum).sum(axis=1).mean()
            self.l_out_endpt = self.l_out_cum
        elif mode == "emd22":
            if self.debug:
                print "train_loss: emd22"
            train_loss = (squared_error(self.net_out_cum, y_cum).sum(axis=1)**2).mean()
            self.l_out_endpt = self.l_out_cum
        elif mode == "xemd2":
            if self.debug:
                print "train_loss: x-ent + emd2"
            assert "emd2_lambda" in args
            train_loss = categorical_crossentropy(self.net_out, y).mean() + \
              args["emd2_lambda"]*squared_error(self.net_out_cum, y_cum).sum(axis=1).mean()
            self.l_out_endpt = self.l_out_cum
        elif mode == "exp":
            if self.debug:
                print "train_loss: exp"
            train_loss = squared_error(self.net_out_exp, y_int.dimshuffle(0,'x')).mean()
            self.l_out_endpt = self.l_out_exp
        elif mode == "qwk":
            if self.debug:
                print "train_loss: qwk"
            train_loss = helpers.qwk(self.net_out, y, num_classes=self.num_classes)
            self.l_out_endpt = self.l_out
        if "l2" in args:
            print "applying l2: %f" % args["l2"]
            train_loss += args["l2"]*regularize_network_params(self.l_out, l2)
        loss_xent = categorical_crossentropy(self.net_out_det, y).mean()
        loss_emd = squared_error(self.net_out_cum_det, y_cum).mean()
        grads = T.grad(train_loss, self.params)
        updates = optimiser(grads, self.params, **optimiser_args)
        train_fn = theano.function(inputs=[X, y, y_ord, y_cum, y_int], outputs=[train_loss, loss_xent], updates=updates,
                                   on_unused_input='warn')
        xent_fn = theano.function(inputs=[X, y], outputs=loss_xent)
        emd_fn = theano.function(inputs=[X, y_cum], outputs=loss_emd)
        dists_fn = theano.function(inputs=[X], outputs=[self.net_out_det, self.net_out_cum_det, self.net_out_exp_det])
        self.fns = {
            "train_fn": train_fn,
            "xent_fn": xent_fn,
            "emd_fn": emd_fn,
            "dists_fn": dists_fn
        }

    def load_weights_from(self, filename):
        with open(filename) as g:
            set_all_param_values(self.l_out_endpt, pickle.load(g))

    def save_weights_to(self, filename):
        with open(filename, "wb") as g:
            pickle.dump(get_all_param_values(self.l_out_endpt), g, pickle.HIGHEST_PROTOCOL)

    def _create_results_file(self, out_dir, filename, append):
        out_file = "%s/%s.txt" % (out_dir, filename)
        if append:
            f_out_file = open(out_file,"a")
        else:
            f_out_file = open(out_file, "wb")
        return f_out_file

    def run_for_model(self, data, iterator_fn, batch_size, model_name):
        Xt, yt, Xv, yv = data
        """
        run one-offs for a model name
        """
        pass

    def train(self, data, iterator_fn, batch_size, num_epochs, out_dir, schedule={}, resume=None, save_every=1, save_to=None, rnd_state=np.random.RandomState(0), debug=False):
        assert save_every >= 1
        header = ["epoch", "train_loss", "train_xent", "valid_xent", "valid_emd", "valid_xent_accuracy", "valid_exp_accuracy", "valid_xent_qwk", "valid_exp_qwk", "learning_rate", "time"]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # initialise log files
        fs = {}
        fs["f_out_file"] = self._create_results_file(out_dir, "results", True if resume != None else False)
        fs["f_train_loss"] = self._create_results_file(out_dir, "train_loss", True if resume != None else False)
        fs["f_train_xent"] = self._create_results_file(out_dir, "train_xent", True if resume != None else False)
        fs["f_valid_xent"] = self._create_results_file(out_dir, "valid_xent", True if resume != None else False)
        if resume != None:
            print "loading weights from: %s" % resume
            self.load_weights_from(resume)
        else:
            fs["f_out_file"].write(",".join(header) + "\n")
        print ",".join(header)
        Xt, yt, Xv, yv = data
        train_fn, xent_fn, emd_fn, dists_fn = self.fns["train_fn"], self.fns["xent_fn"], self.fns["emd_fn"], self.fns["dists_fn"]
        for epoch in range(num_epochs):
            t0 = time()
            if epoch+1 in schedule:
                self.learning_rate.set_value( floatX(schedule[epoch+1]) )
            train_losses = []
            train_xent_losses = []
            for Xb, yb in iterator_fn(Xt, yt, bs=batch_size, num_classes=self.num_classes, rnd_state=rnd_state):
                yb_ord, yb_cum, yb_int = helpers.one_hot_to_ord(yb), helpers.one_hot_to_cmf(yb), helpers.one_hot_to_label(yb)
                t1, t2 = train_fn(Xb, yb, yb_ord, yb_cum, yb_int)
                train_losses.append(t1); fs["f_train_loss"].write("%f\n" % t1)
                train_xent_losses.append(t2); fs["f_train_xent"].write("%f\n" % t2)
                if debug:
                    break
            valid_xent_losses = []
            valid_emd_losses = []
            valid_xent_correct = []
            valid_exp_correct = []
            valid_xent_preds = [] # integer labels for argmax
            valid_exp_preds = [] # integer labels for exp
            for Xb, yb in iterator_fn(Xv, yv, bs=batch_size, num_classes=self.num_classes):
                yb_ord, yb_cum, yb_int = helpers.one_hot_to_ord(yb), helpers.one_hot_to_cmf(yb), helpers.one_hot_to_label(yb)
                valid_xent_losses.append( xent_fn(Xb, yb) ); fs["f_valid_xent"].write("%f\n" % valid_xent_losses[-1])
                valid_emd_losses.append( emd_fn(Xb, yb_cum) )
                d_xent, d_cum, d_exp = dists_fn(Xb)
                #pdb.set_trace()
                valid_xent_correct += ( np.argmax(d_xent,axis=1) == yb_int ).tolist()
                valid_exp_correct += ( np.round(d_exp)[:,0] == yb_int ).tolist()
                #print valid_correct
                valid_xent_preds += np.argmax(d_xent,axis=1).tolist()
                valid_exp_preds += [ int(x) for x in np.round(d_exp)[:,0].tolist() ]
                if debug:
                    break
            valid_xent_qwk = helpers.weighted_kappa(actual_rater=yv[:].tolist(), human_rater=valid_xent_preds, num_classes=self.num_classes)
            valid_exp_qwk = helpers.weighted_kappa(actual_rater=yv[:].tolist(), human_rater=valid_exp_preds, num_classes=self.num_classes)
            to_write = "%i,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f" % \
                       (epoch+1,
                        np.mean(train_losses),
                        np.mean(train_xent_losses),
                        np.mean(valid_xent_losses),
                        np.mean(valid_emd_losses),
                        np.mean(valid_xent_correct),
                        np.mean(valid_exp_correct),
                        valid_xent_qwk,
                        valid_exp_qwk,
                        self.learning_rate.get_value(),
                        time()-t0)
            fs["f_out_file"].write("%s\n" % to_write)
            print to_write
            for key in fs:
                fs[key].flush()
            if save_to != None:
                if epoch % save_every == 0:
                    self.save_weights_to("%s.modelv1.%i" % (save_to, epoch + 1))

    def dump_dists(self, X, y, iterator_fn, batch_size, out_file):
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
