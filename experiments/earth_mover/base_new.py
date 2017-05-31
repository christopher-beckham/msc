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
from keras_ports import ReduceLROnPlateau

def dummy_net_fn(args):
    layer = InputLayer((None, 1, 28, 28))
    layer = DenseLayer(layer, num_units=5)
    return layer

def test_iterator(data, iterator_fn, num_classes):
    Xt, yt, Xv, yv = data
    for Xb, yb in iterator_fn(Xt, yt, bs=batch_size, num_classes=num_classes):
        yield Xb, yb

def sigm_scaled(k):
    def fn(x):
        return sigmoid(x)*(k-1)
    return fn
        
class NeuralNet():

    def print_network(self, l_out):
        for layer in get_all_layers(l_out):
            print layer, layer.output_shape, " " if not hasattr(layer, 'nonlinearity') else layer.nonlinearity

    def _pmf_to_cmf(self, l_out):
        return layers.UpperRightOnesLayer(l_out)

    def _pmf_to_exp(self, l_out):
        l_exp = DenseLayer(l_out, num_units=1, nonlinearity=linear)
        mat = np.asarray([[i] for i in range(self.num_classes)]).astype("float32")
        l_exp.W.set_value(mat)
        return l_exp
    
    def _pmf_to_sq_err(self, l_out, nonlinearity):
        l_sq = DenseLayer(l_out, num_units=1, nonlinearity=nonlinearity)
        return l_sq

    """
    def _get_mode(mode, l_out, ys):
        #:mode: what mode are we training in
        #:l_out: this is a probability distribution layer
        #:ys: this is a tuple of symbolic y values, e.g. y, y_ord, y_cum, etc...
        #:returns: each inner function modifies a dict of the form (which gets returned by the outer function):
        #  train_loss: training loss expression to minimise
        #  xent_loss: x-entropy loss expression (not deterministic)
        #  dists: returns expressions for probability dist and exp dist (deterministic)
        #  last_layer: the very last layer, for model saving
        assert l_out.nonlinearity == softmax
        return_dict = {"train_loss":None, "xent_loss":None, "dists":None, "last_layer":None}
        y, y_ord, y_cum, y_int, y_soft = ys
        def x_ent():
            if self.debug:
                print "train_loss: xent"
            dist_out, dist_out_det = get_output(l_out, X), get_output(l_out, X, deterministic=True) # p(y|x)
            l_exp = self._pmf_to_exp(l_out)
            exp_out, exp_out_det = get_output(l_exp, X), get_output(l_exp, X, deterministic=True) # p(c|x)
            train_loss = categorical_crossentropy(dist_out, y).mean()
            loss_xent = train_loss
            return_dict["train_loss"] = train_loss
            return_dict["xent_loss"] = loss_xent
            return_dict["dists"] = [dist_out_det, exp_out_det]
            return_dict["last_layer"] = l_out
        def emd2():
            if self.debug:
                print "train_loss: emd2"
            l_out_cum = self._pmf_to_cmf(l_out)
            net_out_cum, net_out_cum_det = get_output(l_out_cum, X), get_output(l_out_cum, X, deterministic=True) # cumulative p(y|x)
            train_loss = squared_error(net_out_cum, y_cum).sum(axis=1).mean()
            dist_out, dist_out_det = get_output(l_out, X), get_output(l_out, X, deterministic=True) # p(y|x)
            l_exp = self._pmf_to_exp(l_out)
            exp_out, exp_out_det = get_output(l_exp, X), get_output(l_exp, X, deterministic=True) # p(c|x)
            return_dict["train_loss"] = train_loss
            return_dict["xent_loss"] = categorical_crossentropy(dist_out, y).mean()
            return_dict["dists"] = [dist_out_det, exp_out_det]
            return_dict["last_layer"] = l_out_cum
        def exp():
            if self.debug:
                print "train_loss: exp"
            l_out_exp = self._pmf_to_exp(l_out)
            net_out_exp, net_out_exp_det = get_output(l_out_exp, X), get_output(l_out_exp, X, deterministic=True)
            train_loss = squared_error(net_out_exp, y_int.dimshuffle(0,'x')).mean()
            dist_out, dist_out_det = get_output(l_out, X), get_output(l_out, X, deterministic=True)
            dists_fn = theano.function([X], [dist_out_det, net_out_exp_det])
            return_dict["train_loss"] = train_loss
            return_dict["xent_loss"] = categorical_crossentropy(dist_out, y).mean()
            return_dict["dists"] = [dist_out_det, exp_out_det]
            return_dict["last_layer"] = l_out_exp        
            
        fn_to_return = {
            "x_ent": x_ent,
            "emd2":emd2,
            "exp":exp
        }
        fn_to_return[mode]()
        
        for key in return_dict:
            assert return_dict[key] != None

        return return_dict
    """
        
    def __init__(self, net_fn, num_classes, optimiser=nesterov_momentum,
                     optimiser_args={"learning_rate":theano.shared(floatX(0.01)),"momentum":0.9}, mode="x_ent", args={}, debug=False):
        assert mode in \
            ["x_ent", "soft", "emd2", "emd22", "xemd2", "exp", "qwk", "sq_err", "sq_err_classic", "qwk_reform", "qwk_reform_classic", "sq_err_backrelu", "sq_err_classic_backrelu", "sq_err_fx"]
        #TODO: clean backrelu
        self.num_classes = num_classes
        self.learning_rate = optimiser_args["learning_rate"]
        self.debug = debug
        # this MUST return a probability distribution of some sort, even
        # if it's a dummy distribution
        l_out = net_fn(args)
        assert l_out.output_shape[-1] == self.num_classes
        self.print_network(l_out)
        self.l_in = get_all_layers(l_out)[0]
        self.l_out_endpt = None
        # theano variables
        X = T.tensor4('X'); self.input_tensor = X
        y = T.fmatrix('y')
        # TODO: this is not very clean
        y_ord, y_cum, y_int, y_soft = T.fmatrix('y_ord'), T.fmatrix('y_cum'), T.ivector('y_int'), T.fmatrix('y_soft')
        self.y_soft_sigma = 1.
        # ---
        self.params = get_all_params(l_out, trainable=True)
        if self.debug:
            print "params: ", self.params
        if self.debug:
            print "train_loss: %s" % mode
        if mode == "x_ent":
            dist_out, dist_out_det = get_output(l_out, X), get_output(l_out, X, deterministic=True) # p(y|x)
            l_exp = self._pmf_to_exp(l_out)
            exp_out, exp_out_det = get_output(l_exp, X), get_output(l_exp, X, deterministic=True) # p(c|x)
            dists_fn = theano.function([X], [dist_out_det, exp_out_det]) # get p(y|x) and p(c|x)
            train_loss = categorical_crossentropy(dist_out, y).mean()
            self.l_out_endpt = l_out
            #self.tmp_fn = theano.function([X], get_output([l_out, l_out.input_layer, l_out.input_layer.input_layer, l_out.input_layer.input_layer.input_layer], X, deterministic=True))
        elif mode == "soft":
            """
            if self.debug:
                print "train_loss: soft"
            train_loss = categorical_crossentropy(self.net_out, y_soft).mean()
            self.l_out_endpt = self.l_out
            self.y_soft_sigma = args["y_soft_sigma"]
            """
            raise NotImplementedError()
        elif mode == "emd2":
            l_out_cum = self._pmf_to_cmf(l_out)
            net_out_cum, net_out_cum_det = get_output(l_out_cum, X), get_output(l_out_cum, X, deterministic=True) # cumulative p(y|x)
            train_loss = squared_error(net_out_cum, y_cum).sum(axis=1).mean()
            dist_out, dist_out_det = get_output(l_out, X), get_output(l_out, X, deterministic=True) # p(y|x)
            l_exp = self._pmf_to_exp(l_out)
            exp_out, exp_out_det = get_output(l_exp, X), get_output(l_exp, X, deterministic=True) # p(c|x)
            dists_fn = theano.function([X], [dist_out_det, exp_out_det]) # get p(y|x) and p(c|x)
            self.l_out_endpt = l_out_cum
        elif mode == "emd22":
            """
            if self.debug:
                print "train_loss: emd22"
            train_loss = (squared_error(self.net_out_cum, y_cum).sum(axis=1)**2).mean()
            self.l_out_endpt = self.l_out_cum
            """
            raise NotImplementedError()
        elif mode == "xemd2":
            """
            if self.debug:
                print "train_loss: x-ent + emd2"
            assert "emd2_lambda" in args
            train_loss = categorical_crossentropy(self.net_out, y).mean() + \
              args["emd2_lambda"]*squared_error(self.net_out_cum, y_cum).sum(axis=1).mean()
            self.l_out_endpt = self.l_out_cum
            """
            raise NotImplementedError()
        elif mode == "exp":
            l_out_exp = self._pmf_to_exp(l_out)
            net_out_exp, net_out_exp_det = get_output(l_out_exp, X), get_output(l_out_exp, X, deterministic=True)
            train_loss = squared_error(net_out_exp, y_int.dimshuffle(0,'x')).mean()
            dist_out, dist_out_det = get_output(l_out, X), get_output(l_out, X, deterministic=True)
            dists_fn = theano.function([X], [dist_out_det, net_out_exp_det])
            self.l_out_endpt = l_out_exp
        elif mode in ["sq_err", "sq_err_backrelu", "sq_err_classic", "sq_err_classic_backrelu"]:
            # TODO: don't put the relu in the option name, it is kinda unclean
            
            # sq_err = softmax hidden layer, linear exp layer
            # sq_err_classic = softmax hidden layer, scaled sigm exp layer
            # sq_err_backrelu = relu hidden layer, linear exp layer
            # sq_err_classic_backrelu = relu hidden layer, scaled sigm exp layer
            is_classic = True if "classic" in mode else False
            if "backrelu" in mode:
                l_out.nonlinearity = rectify
            l_out_exp = self._pmf_to_sq_err(l_out, nonlinearity=sigm_scaled(self.num_classes) if is_classic else rectify)
            net_out_exp, net_out_exp_det = get_output(l_out_exp, X), get_output(l_out_exp, X, deterministic=True)
            train_loss = squared_error(net_out_exp, y_int.dimshuffle(0,'x')).mean()
            dist_out, dist_out_det = get_output(l_out, X), get_output(l_out, X, deterministic=True)
            dists_fn = theano.function([X], [dist_out_det, net_out_exp_det])
            self.l_out_endpt = l_out_exp
        elif mode == "sq_err_fx":
            # TODO: FIXXXX
            # HACKY: this applies squared_error to a layer before l_out
            # whose tag is 'fx'. This was intended to applied in
            # conjunction with the Poisson extension
            l_fx = None
            for layer in get_all_layers(l_out)[::-1]:
                if hasattr(layer, 'tag') and layer.tag == 'fx':
                    l_fx = layer
                    break
            if l_fx == None:
                raise Exception("For sq_err_fx, a layer must have the tag 'fx' associated with it.")
            net_out_fx, net_out_fx_det = get_output(l_fx, X), get_output(l_fx, X, deterministic=True)
            #l_out_exp = self._pmf_to_sq_err(l_out)
            #net_out_exp = get_output(l_out_exp, X, deterministic=True)
            net_out_exp = net_out_fx
            train_loss = squared_error(net_out_fx, y_int.dimshuffle(0,'x')).mean()
            dist_out, dist_out_det = get_output(l_out, X), get_output(l_out, X, deterministic=True)
            dists_fn = theano.function([X], [dist_out_det, net_out_fx_det])
            self.l_out_endpt = l_out
        elif mode == "qwk":
            l_out_exp = self._pmf_to_exp(l_out)
            net_out_exp, net_out_exp_det = get_output(l_out_exp, X), get_output(l_out_exp, X, deterministic=True)
            dist_out, dist_out_det = get_output(l_out, X), get_output(l_out, X, deterministic=True)
            train_loss = helpers.qwk(dist_out, y, num_classes=self.num_classes)
            dists_fn = theano.function([X], [dist_out_det, net_out_exp_det])
            self.l_out_endpt = l_out
        elif mode in ["qwk_reform", "qwk_reform_classic"]:
            # qwk_reform = reformulated qwk as a regression, not using sigmoid trick
            # qwk_reform_classic = "" but using sigmoid trick
            l_out.nonlinearity = rectify # OVERRIDE nonlinearity
            l_out_exp = self._pmf_to_sq_err(l_out, nonlinearity=sigm_scaled(self.num_classes) if mode == "qwk_reform_classic" else rectify)
            net_out_exp, net_out_exp_det = get_output(l_out_exp, X), get_output(l_out_exp, X, deterministic=True)
            train_loss = helpers.qwk_reform_fixed(net_out_exp, y_int)
            self.debug_tps = helpers.qwk_num_denom(net_out_exp, y_int)
            self.debug_fn = theano.function([X,y_int], self.debug_tps)
            dist_out, dist_out_det = get_output(l_out, X), get_output(l_out, X, deterministic=True)
            dists_fn = theano.function([X], [dist_out_det, net_out_exp_det])
            self.l_out_endpt = l_out_exp
        if "l2" in args:
            print "applying l2: %f" % args["l2"]
            # BUG: this does not affect the l_exp layer??
            train_loss += args["l2"]*regularize_network_params(l_out, l2)
        # get monitors
        monitors = {}
        for layer in get_all_layers(self.l_out_endpt):
            if layer.name != None:
                monitors[layer.name] = theano.function([X], get_output(layer, X))
        print "monitors found:", monitors.keys()
        loss_xent = categorical_crossentropy(dist_out, y).mean()
        #loss_emd = squared_error(self.net_out_cum_det, y_cum).mean()
        grads = T.grad(train_loss, self.params)
        updates = optimiser(grads, self.params, **optimiser_args)
        train_fn = theano.function(inputs=[X, y, y_ord, y_cum, y_int, y_soft], outputs=[train_loss, loss_xent], updates=updates,
                                   on_unused_input='warn')
        loss_fn = theano.function(inputs=[X, y, y_ord, y_cum, y_int, y_soft], outputs=train_loss, on_unused_input='warn')
        xent_fn = theano.function(inputs=[X, y], outputs=loss_xent)
        #emd_fn = theano.function(inputs=[X, y_cum], outputs=loss_emd)
        #dists_fn = theano.function(inputs=[X], outputs=[self.net_out_det, self.net_out_cum_det, self.net_out_exp_det])
        # DEBUG purposes
        self.train_loss = train_loss
        self.X = X
        self.y = y
        # ----
        self.fns = {
            "train_fn": train_fn,
            "loss_fn": loss_fn,
            "xent_fn": xent_fn,
            "dists_fn": dists_fn
        }
        self.monitors = monitors

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

    def _plot_network(self, l_out, out_file):
        from nolearn.lasagne.visualize import draw_to_file
        draw_to_file( get_all_layers(l_out), out_file, verbose=True )
        
    def train(self, data, iterator_fn, batch_size, num_epochs, out_dir, schedule={}, resume=None, save_every=1, save_to=None,
              rnd_state=np.random.RandomState(0), reduce_lr_on_plateau=None, debug=False, pdb_debug=False):
        assert save_every >= 1
        header = ["epoch",
                  "train_loss",
                  "train_xent",
                  "valid_loss",
                  "valid_xent",
                  "valid_xent_accuracy",
                  "valid_exp_accuracy",
                  "valid_xent_qwk",
                  "valid_exp_qwk",
                  "learning_rate", "mb_time", "time"]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # initialise log files
        fs = {}
        fs["f_out_file"] = self._create_results_file(out_dir, "results", True if resume != None else False)
        #fs["f_train_loss"] = self._create_results_file(out_dir, "train_loss", True if resume != None else False)
        #fs["f_train_xent"] = self._create_results_file(out_dir, "train_xent", True if resume != None else False)
        #fs["f_valid_xent"] = self._create_results_file(out_dir, "valid_xent", True if resume != None else False)
        fs_monitors = {}
        for key in self.monitors:
            fs_monitors[key] = self._create_results_file(out_dir, key, True if resume != None else False)
        if resume != None:
            print "loading weights from: %s" % resume
            self.load_weights_from(resume)
        else:
            fs["f_out_file"].write(",".join(header) + "\n")
        print ",".join(header)
        print "learning rate schedule:", schedule
        self._plot_network(self.l_out_endpt, "%s/network.png" % out_dir)
        Xt, yt, Xv, yv = data
        train_fn, loss_fn, xent_fn, dists_fn = \
                self.fns["train_fn"], self.fns["loss_fn"], self.fns["xent_fn"], self.fns["dists_fn"]
        lr_reducer = ReduceLROnPlateau(self.learning_rate, verbose=1)
        if reduce_lr_on_plateau != None:
            assert reduce_lr_on_plateau in ['valid_loss', 'valid_exp_qwk']
            lr_reducer.on_train_begin()
        for epoch in range(num_epochs):
            t0 = time()
            if epoch+1 in schedule:
                self.learning_rate.set_value( floatX(schedule[epoch+1]) )
            train_losses, train_xent_losses, mb_time = [], [], []
            for Xb, yb in iterator_fn(Xt, yt, bs=batch_size, num_classes=self.num_classes, rnd_state=rnd_state):
                yb_ord, yb_cum, yb_int, yb_soft = \
                    helpers.one_hot_to_ord(yb), helpers.one_hot_to_cmf(yb), helpers.one_hot_to_label(yb), helpers.one_hot_to_soft(yb,self.y_soft_sigma)
                mb_t0 = time()
                if pdb_debug:
                    import pdb
                    pdb.set_trace()
                t1, t2 = train_fn(Xb, yb, yb_ord, yb_cum, yb_int, yb_soft)
                #print t1
                mb_time.append( time()-mb_t0 )
                train_losses.append(t1); #fs["f_train_loss"].write("%f\n" % t1)
                train_xent_losses.append(t2); #fs["f_train_xent"].write("%f\n" % t2)
                for key in fs_monitors:
                    fs_monitors[key].write("%s\n" % self.monitors[key](Xb).flatten().tolist())
                if debug:
                    break
            valid_losses, valid_xent_losses, valid_emd_losses, valid_xent_correct, valid_exp_correct, valid_xent_preds, valid_exp_preds = \
                [], [], [], [], [], [], []
            for Xb, yb in iterator_fn(Xv, yv, bs=batch_size, num_classes=self.num_classes):
                yb_ord, yb_cum, yb_int, yb_soft = \
                    helpers.one_hot_to_ord(yb), helpers.one_hot_to_cmf(yb), helpers.one_hot_to_label(yb), helpers.one_hot_to_soft(yb,self.y_soft_sigma)
                valid_losses.append( loss_fn(Xb, yb, yb_ord, yb_cum, yb_int, yb_soft) )
                valid_xent_losses.append( xent_fn(Xb, yb) ); #fs["f_valid_xent"].write("%f\n" % valid_xent_losses[-1])
                #valid_emd_losses.append( emd_fn(Xb, yb_cum) )
                d_xent, d_exp = dists_fn(Xb)
                #pdb.set_trace()
                valid_xent_correct += ( np.argmax(d_xent,axis=1) == yb_int ).tolist()
                valid_exp_correct += ( np.clip(np.round(d_exp)[:,0], 0, self.num_classes-1) == yb_int ).tolist()
                #print valid_correct
                valid_xent_preds += np.argmax(d_xent,axis=1).tolist()
                valid_exp_preds += [ int(x) for x in np.round(d_exp)[:,0].tolist() ]
                if debug:
                    break
            #print np.mean(mb_time)
            valid_xent_qwk = helpers.weighted_kappa(actual_rater=yv[:].tolist(), human_rater=valid_xent_preds, num_classes=self.num_classes)
            valid_exp_qwk = helpers.weighted_kappa(actual_rater=yv[:].tolist(), human_rater=valid_exp_preds, num_classes=self.num_classes)
            to_write = "%i,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f" % \
                       (epoch+1,
                        np.mean(train_losses),
                        np.mean(train_xent_losses),
                        np.mean(valid_losses),
                        np.mean(valid_xent_losses),
                        np.mean(valid_xent_correct),
                        np.mean(valid_exp_correct),
                        valid_xent_qwk,
                        valid_exp_qwk,
                        self.learning_rate.get_value(),
                        np.mean(mb_time),
                        time()-t0)
            fs["f_out_file"].write("%s\n" % to_write)
            print to_write
            for key in fs:
                fs[key].flush()
            for key in fs_monitors:
                fs_monitors[key].flush()
            if save_to != None:
                if epoch % save_every == 0:
                    self.save_weights_to("%s.modelv1.%i" % (save_to, epoch + 1))
            if reduce_lr_on_plateau != None:
                if reduce_lr_on_plateau == 'valid_loss':
                    lr_reducer.on_epoch_end( np.mean(valid_losses) , epoch+1 )
                else:
                    lr_reducer.on_epoch_end( valid_exp_qwk, epoch+1 )

    def dump_dists(self, X, y, iterator_fn, batch_size, out_file, rnd_state=np.random.RandomState(0), debug=False):
        """
        :param X: input data
        :param out_file: p(y|x)
        :return:
        """
        print rnd_state.randint(0,1000)
        with open(out_file,"wb") as f:
            ctr = 0
            for Xb, yb in iterator_fn(X, y, bs=batch_size, num_classes=self.num_classes, rnd_state=rnd_state):
                ctr += 1
                if debug:
                    print "processing batches %i to %i" % (ctr*batch_size, (ctr+1)*batch_size)
                dists, _ = self.fns["dists_fn"](Xb)
                for i, row in enumerate(dists):
                    row = [ str(elem) for elem in row.tolist() ]
                    row.append(str(np.argmax(yb[i])))
                    f.write(",".join(row) + "\n")

    def dump_output_for_layer(self, layer, X, y, iterator_fn, batch_size, out_file):
        print "dumping output for layer %s with output shape %s" % (str(layer), str(layer.output_shape))
        with open(out_file, "wb") as f:
            layer_out = get_output(layer, self.input_tensor)
            layer_out_fn = theano.function([self.input_tensor], layer_out)
            for Xb, yb in iterator_fn(X, y, bs=batch_size, num_classes=self.num_classes):
                # e.g. (bs, p)
                out_for_this_batch = layer_out_fn(Xb)
                for i, row in enumerate(out_for_this_batch):
                    row = row.flatten().tolist()
                    row = [ str(elem) for elem in row ]
                    row.append(str(np.argmax(yb[i])))
                    f.write(",".join(row) + "\n")

if __name__ == '__main__':
    pass
