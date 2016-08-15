import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from datetime import datetime
import os



FILES = [sys.argv[1]]


read = []
for F in FILES:
    with open(F) as f:
        read.append(f.read())

iterations = [r.split("solver.cpp:228]")[1:] for r in read]


frames = []
for ix,iteration in enumerate(iterations):
    fig,ax=plt.subplots(figsize=(10,6))

    rr = {"Iteration" : re.compile("Iteration\s(\d+)\,"),
          "loss"      : re.compile("\,\sloss\s\=\s(\d+\.\d+(e\-\d+)?)\s"),
          "acc"       : re.compile("accuracy\s\=\s(\d+\.*\d*(e\-\d+)?)\s")}
    di = {"Iteration" : [],
          "loss" : [],
          "acc" : []}

    for i in iteration:
        for r in rr:
            a = rr[r].search(i)
            if a is None: 
                continue
            a = a.group(1)   
            a = float(a)
            di[r].append(a)

    # converting to series means pandas will be O.K. with unequal length
    # lists
    for key,val in di.iteritems():
        di[key] = pd.Series(val)

    df = pd.DataFrame.from_dict(di)   
    #############################
    frames.append(df)
    print df.shape
    matplotlib.rcParams['font.size'] = 16
    
    batch = float(sys.argv[2])
    ntotal = 2*105786

    ax.plot(df.Iteration.values*batch/ntotal,
            df.loss.rolling(window=1).mean(),'-',color='blue',lw=2,label='loss',alpha=0.6)
    ax.plot(df.Iteration.values*batch/ntotal,
            df.loss.rolling(window=10).mean(),'-',color='red',lw=2,label='avg')

    #ax2 = ax.twinx()
    #ax2.plot(df.Iteration.values*batch/ntotal,
    #         df.acc.rolling(window=1).mean(),'-',label='acc',color='green',lw=2,alpha=0.6)
    #ax2.plot(df.Iteration.values*batch/ntotal,
    #         df.acc.rolling(window=10).mean(),'-',label='avg',color='orange',lw=2)

    #ax.legend(loc='best')
    #ax2.plot(np.array([0.0,100.,200.,300.])*batch/ntotal,
    #                 np.array([.830,.760, .770, .690]),'-o',color='purple',lw=2)

    ax.set_xlabel("Epoch",fontweight='bold')
    ax.set_ylabel("Loss",fontweight='bold')
    #ax.set_ylim(0.0,2.0)
    #ax2.set_ylim(0.0,1.0)
    #ax2.set_ylabel("Accuracy",fontweight='bold')
    ax.legend(loc='lower left')
    #ax2.legend(loc='lower right')
    ax.set_title("{}\n".format(FILES[ix].split("/")[-1].split(".")[0]))
    #ax.set_xlim(0,0.6)
    #plt.savefig('train_log_{}.png'.format(ix), format='png', dpi=500)
    plt.grid()
    ax.set_yscale("log")
    plt.savefig("aho_out_loss.png",type='png')
