import caffe
import numpy as np
import ROOT as rt
import lmdb
import time
from ROOT import larcv
caffe.set_mode_gpu()
caffe.set_device(2)

PROTO = "ana.prototxt"
MODEL = "weights/fcn_iter_1500.caffemodel"
BATCH_CTR=1
ANA_OUTPUT_CFG="ana_out.cfg"

proc = larcv.ProcessDriver('ProcessDriver')

proc.configure(ANA_OUTPUT_CFG)

proc.initialize()

py_image_maker = proc.process_ptr(proc.process_id("PyImageMaker"))

outman = larcv.IOManager(larcv.IOManager.kWRITE)

net = caffe.Net( PROTO, MODEL, caffe.TEST)

filler = larcv.ThreadFillerFactory.get_filler("Ana")

num_events = filler.get_n_entries()

filler.set_next_index(0)

print
print 'Total number of events:',num_events
print 'Batch size:', BATCH_CTR
print

current_index = 0
last_index = 0
for ibatch in range(0,num_events / BATCH_CTR+1):

    last_index = current_index    

    current_index = ibatch * BATCH_CTR

    print 'Batch:',ibatch
    if not last_index/1000 == current_index/1000:
        print "Current index @",current_index

    num_entries = num_events - current_index

    if num_entries > BATCH_CTR: num_entries = BATCH_CTR

    net.forward()

    labels  = net.blobs["label"].data
    softmax = net.blobs["loss_"].data

    while filler.thread_running():
        time.sleep(0.001)

    events = filler.processed_events()
    entries = filler.processed_entries()

    if events.size() != BATCH_CTR:
        print "Batch counter mis-match!"
        raise Exception

    print labels.shape
    print softmax.shape
    print softmax.dtype
    
    for idx in xrange(num_entries):

        img_array = softmax[idx]
        for ch in xrange(len(img_array)):
            img = img_array[ch]
            py_image_maker.append_ndarray(img.transpose())
        py_image_maker.set_id(events[idx].run(),events[idx].subrun(),events[idx].event())
        proc.process_entry()

    if num_entries < BATCH_CTR:
        break
proc.finalize()

