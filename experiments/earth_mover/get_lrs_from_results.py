
def get_lrs(filename):
    curr_lr = None
    num_epochs = 1
    results = {}
    try:
        with open(filename) as f:
            header = f.readline().split(",")
            lr = header.index('learning_rate')
            for line in f:
                line = line.rstrip().split(",")
                this_lr = float(line[lr])
                if curr_lr == None:
                    curr_lr = this_lr
                if this_lr != curr_lr:
                    curr_lr = this_lr
                    #print "lr changed to %f at epoch %i" % (curr_lr, num_epochs)
                    results[num_epochs] = curr_lr
                num_epochs += 1
        print "get_lrs(%s) =" % filename, results
        return results
    except:
        # if file don't exist then return a blank sched
        return results

if __name__ == '__main__':
    #sched = get_lrs("output/adience_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")
    #print sched
    sched = get_lrs("/tmp/blah.txt")
    print sched
