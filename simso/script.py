#import config_test_1 as cc
#import pandas as pd
#
#c2=[{'task':'adpcm', 'cpu':1, 'period': 10},{'task':'compress', 'cpu':0, 'period': 5},{'task':'edn', 'cpu':0, 'period': 15},{'task':'crc', 'cpu':0, 'period': 10},{'task':'ndes', 'cpu':1, 'period': 15}]
#c3=[{'task':'adpcm', 'cpu':1, 'period': 10},{'task':'compress', 'cpu':0, 'period': 5},{'task':'edn', 'cpu':0, 'period': 15},{'task':'crc', 'cpu':1, 'period': 10},{'task':'ndes', 'cpu':1, 'period': 15}]
#for k,j in enumerate([c2, c3]):
#    for i in range(1):
#        res, load = cc.exp2(j, 5)
#        pd.to_pickle(res, 'resc'+str(k+2)+str(i)+'.pickle')
#        pd.to_pickle(load, 'util'+str(k+2)+str(i)+'.pickle')


import config_test_1 as cc
import pandas as pd
import re, os, shutil, glob

c2=[{'task':'adpcm', 'cpu':1, 'period': 10, 'act_d':0},{'task':'compress', 'cpu':0, 'period': 5, 'act_d':0},{'task':'edn', 'cpu':0, 'period': 15, 'act_d':0},{'task':'crc', 'cpu':0, 'period': 10, 'act_d':0},{'task':'ndes', 'cpu':1, 'period': 15, 'act_d':0}]

#scheds=['G_FL', 'EDZL', 'EKG', 'EDHS', 'EDF']
#scheds=['LLREF', 'LRE-DL', 'DP-WRAP', 'BF', 'NVNLF', 'U-EDF']
#scheds=['MLLF', 'PriD', 'RM']
scheds=['G_FL', 'EKG', 'EDF', 'RM', 'PriD']
caches=[16, 32, 64, 128, 256, 512]

for c in caches:
    for sch in scheds:
        # file name
        name = 'res'+sch+str(c)+'.pickle'

        # check if the task is being or has been executed
        f = name
        f0 = name + '.tmp'
        if (os.path.isfile(f) | os.path.isfile(f0)):
            print 'File {0} exists.'.format(f)
            continue

        # create the tmp file
        open(f0, 'a').close()
        try:
            res = cc.exp3([sch], c2, [c])

            pd.to_pickle(res, f)

            # remove .tmp file
            os.remove(f0)
         
        except:
            
            print 'xxxxxxxxxxxxxxxxxxx:', sch
