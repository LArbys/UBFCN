import sys, os
import caffe
import numpy as np
import ROOT as rt
import lmdb
import time
from ROOT import larcv

NUM_MAX   = 100
PI_THRESH = 50.

def model_flist(prefix):

    if not prefix.startswith('/'): 
        prefix = '%s/%s' % (os.getcwd(),prefix)

    indir  = prefix[0:prefix.rfind('/')]
    prefix = prefix[prefix.rfind('/')+1:len(prefix)]

    flist = [x for x in os.listdir(indir) if x.startswith(prefix) and (x.endswith('.caffemodel') or x.endswith('.caffemodel.h5'))]

    fmap={}
    for f in flist:
        ftmp = f.replace(prefix,'')
        ftmp = ftmp.replace('.caffemodel.h5','')
        ftmp = ftmp.replace('.caffemodel','')
        if not ftmp.isdigit(): continue
        fmap[int(ftmp)] = '%s/%s' % (indir,f)
    return fmap

#
# Get iteration + weight file list
#

PREFIX = sys.argv[2]
fmap = model_flist(PREFIX)
iters = fmap.keys()
iters.sort()
#for i in iters: print i,fmap[i]

#
# Construct net
#
caffe.set_mode_gpu()
if len(sys.argv) > 3:
    caffe.set_device(int(sys.argv[3]))

NETCFG = sys.argv[1]
BATCH_CTR=0
for w in open(NETCFG,'r').read().replace(':',' ').split():
    if BATCH_CTR is None:
        BATCH_CTR = int(w)
        break
    if w == 'batch_size':
        BATCH_CTR = None
if not BATCH_CTR:
    print 'Error: failed to get batch_ctr from prototxt...'
    sys.exit(1)

#
# Retrieve 
#
inference_fname = NETCFG.replace('.prototxt','.txt')
inference_fname = "inference_%s" % inference_fname
for iter_num in iters:

    res = {}
    if os.path.isfile(inference_fname):
        for l in [ x for x in open(inference_fname,'r').read().split('\n') if len(x.split())==2 and x.split()[0].isdigit() ]:
            i,p = l.split()
            res[int(i)] = float(p)

    if iter_num in res: continue

    model = fmap[iter_num]
    
    net = caffe.Net( NETCFG, model, caffe.TEST)

    filler = larcv.ThreadFillerFactory.get_filler("Ana")

    num_events = filler.get_n_entries()

    print "num_events:",num_events
    if NUM_MAX and num_events > NUM_MAX : num_events = NUM_MAX

    while filler.thread_running():
        time.sleep(0.001)

    filler.set_next_index(0)

    print
    print 'Total number of events:',num_events
    print 'Batch size:', BATCH_CTR
    print

    good_ctr = 0
    tot_ctr = 0
    current_index = 0
    last_index = 0
    num_batch = num_events / BATCH_CTR + 1
    for ibatch in xrange(num_batch):

        last_index = current_index    

        current_index = ibatch * BATCH_CTR

        current_prob = 0.
        if tot_ctr: current_prob = float(good_ctr) / float(tot_ctr)
        sys.stdout.write('Iteration %-4d Current prob = %g @ batch %-2d/%-2d ...\r' % (iter_num,current_prob,ibatch,num_batch))
        sys.stdout.flush()

        num_entries = num_events - current_index

        if num_entries > BATCH_CTR: num_entries = BATCH_CTR

        net.forward()

        imgs = net.blobs["data"].data
        labels = net.blobs["label"].data
        scores = net.blobs["score"].data
        
        for index in xrange(len(scores)):
            
            img = imgs[index]
            label = labels[index]
            score = scores[index].argmax(axis=0)
            score = score.reshape([ img.shape[0] ] + list(score.shape) )

            thresh_img =  img > PI_THRESH
            npxcorrect = (label[thresh_img] == score[thresh_img]).sum()
            npxthresh  = thresh_img.sum()
            #good_ctr += npxcorrect / float(npxthresh) 
            #tot_ctr  += 1.0
            good_ctr += float(npxcorrect)
            tot_ctr  += float(npxthresh)

    res[iter_num] = good_ctr / tot_ctr

    print 'Iteration %-4d ... Accuracy %g           ' % (iter_num, res[iter_num])
    iters=res.keys()
    iters.sort()
    fout = open(inference_fname,'w')
    for i in iters:
        fout.write('%d %g\n' % (i,res[i]))
    fout.close()

    larcv.ThreadFillerFactory.destroy_filler("Ana")



