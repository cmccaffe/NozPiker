import tensorflow as tf
from tensorflow import keras
import pickle
import os
import numpy as np
from NozPiker_Funcs import main as NZ
import pandas as pd
from biopandas.pdb import PandasPdb
import subprocess
import os
import glob

DATA_DIR = os.path.join(os.path.dirname(os.getcwd()),'data/build_data/')

files = glob.glob(DATA_DIR + "*.mrcs")
madefiles = []
for i in files:
    shortname = i.split('/')[-1].split('_')[1]
    madefiles.append(shortname)

rcsbdata = pd.read_table('/home/cns-mccafferty/NeuralNetFun/cmpd_res.idx', skiprows=4, header=None, names=['pdb', 'a', 'res', 'b', 'compound'])
rcsbdata = rcsbdata[['pdb', 'res', 'compound']]
nonorcsbdata = rcsbdata.dropna() # drops all rows with nan
highres = nonorcsbdata[nonorcsbdata['res'] <= 3] # drops all rows with resolution lower than 3A
reducedhighres = highres.drop_duplicates(subset='compound', keep="first") # only keeps first compound of repeats

# the steps above bring set of initial size 164264 to 128755

# topthous = reducedhighres.head(1000)
pdblist = list(reducedhighres['pdb'])

pdblist.remove('3P4A')
pdblist.remove('5VY7')
pdblist.remove('2C4S')
pdblist.remove('1C4S')
pdblist.remove('1CAP')

l3 = [x for x in pdblist if x not in madefiles]
for i in l3:
    try:
        NZ.makeImages(DATA_DIR, i)
    except AttributeError:
        pass
