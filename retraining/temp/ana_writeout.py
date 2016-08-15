import caffe

caffe.set_mode_gpu()
caffe.set_device(0)

import numpy as np
import ROOT as rt
import lmdb
import time,sys

from ROOT import larcv

MODEL = sys.argv[1]
PROTO = sys.argv[2]
PROTO2=sys.argv[3]

BATCH_CTR=1

MODEL_NAME = MODEL.split("/")

import re

proc = larcv.ProcessDriver('ProcessDriver')

proc.configure(PROTO)

proc.initialize()

py_image_maker = proc.process_ptr(proc.process_id("PyImageMaker"))

net = caffe.Net( PROTO2, MODEL, caffe.TEST)

filler = larcv.ThreadFillerFactory.get_filler("Ana")

while filler.thread_running():
    print "Waiting..."
    time.sleep(0.001)

num_events = filler.get_n_entries()

filler.set_next_index(0)

# calls net.forward() calls
# ROOTDataLayer<Dtype>::~ROOTDataLayer<Dtype>(), with no thread
# shit gets copied, then batch_process_called with set_next_index!
#print "\t=========>Calling forward once\n"
#net.forward() 
#print "\t=========>Called forward once\n"
print
print 'Total number of events:',num_events
print 'Batch size:', BATCH_CTR
print

current_index = 0
last_index = 0

num_events = 10

for ibatch in range(0,num_events / BATCH_CTR):
    
    last_index = current_index    

    current_index = ibatch * BATCH_CTR

    print 'Forward on batch:',ibatch

    num_entries = num_events - current_index

    if num_entries > BATCH_CTR: num_entries = BATCH_CTR

    while filler.thread_running():
        print "Waiting..."
        time.sleep(0.001)
        
    net.forward()

    while filler.thread_running():
        print "Waiting..."
        time.sleep(0.001)        
    
    labels  = net.blobs["label"].data
    score   = net.blobs['score'].data

    events  = filler.processed_events()
    entries = filler.processed_entries()
    
    pd = filler.pd()
    print "filler pd: ",pd.event_id().run(),pd.event_id().subrun(),pd.event_id().event()
    print "proc pd:   ",proc.event_id().run(),proc.event_id().subrun(),proc.event_id().event()
    
    pd_io   = pd.io()
    proc_io = proc.io()
    
    print "pd io():        ",pd_io.event_id().run(),pd_io.event_id().subrun(),pd_io.event_id().event()
    print "pd io() last:   ",pd_io.last_event_id().run(),pd_io.last_event_id().subrun(),pd_io.last_event_id().event()

    print "proc io():      ",proc_io.event_id().run(),proc_io.event_id().subrun(),proc_io.event_id().event()
    print "proc io() last: ",proc_io.last_event_id().run(),proc_io.last_event_id().subrun(),proc_io.last_event_id().event()

    print "entries: ",np.array(entries)

    if events.size() != BATCH_CTR:
        print events.size(),BATCH_CTR,MODEL_NAME
        raise Exception
    
    for idx in xrange(num_entries): # num_entries == batch

        img_array = score[idx]

        for ch in xrange(len(img_array)):
            img = img_array[ch]
            py_image_maker.append_ndarray(img.T)

        print "setting idx: ",idx,events[idx].run(),events[idx].subrun(),events[idx].event()
        py_image_maker.set_id(events[idx].run(),events[idx].subrun(),events[idx].event())

        proc.process_entry()

    print "\n\n"
    if num_entries < BATCH_CTR:
        break

print "Finalize, not saving..."
proc.finalize()

