
# function to sort projected images into dictionary as arrays
def GetImageSet(mrc,store):
    from mrcfile import open
    name = mrc.split('/')[-1].split('_')[0]
    ID = mrc.split('/')[-1].split('_')[1]
    with open(mrc,'r') as m:
        data = []
        for i in range(m.data.shape[0]):
            data.append(m.data[i,:,:])
    size = len(data[-1][:,0])
    store[ID] = [name,data]
    return size

# function to create training data
def CreateTrainingData(store, size):
    from numpy import array, where, zeros, min, max, resize, transpose, floor
    from tensorflow_core.image import resize_with_crop_or_pad
    # Get Dict keys
    class_names = array(sorted(store.keys()))
    # Pre-allocate arrays
    key = class_names[0]
    length = len(store[key][1])
    shape = (size,size)
    train_images = zeros((int(length*(floor(len(class_names)*8/10+(len(class_names)%5)))),shape[0],shape[1]), dtype='uint8')
    train_labels = zeros(int(length*(floor(len(class_names)*8/10+(len(class_names)%5)))))
    test_images = zeros((int(length*(floor(len(class_names)*2/10))),shape[0],shape[1]), dtype='uint8')
    test_labels = zeros(int(length*(floor(len(class_names)*2/10))))
    # Load images and labels
    counter = 0
    j = 0
    k = 0
    names = []
    for key in class_names:
        prot = store[key][0]
        if prot not in names:
            names.append(prot)
    names = array(names)
    for key in class_names:
        counter += 1
        prot = store[key][0]
        loc = where(names == prot)[0]
        # Add images
        if store[key][1][0].shape[1] < size:
                if counter%5 == 0:
                    for i in range(len(store[key][1])):
                        array = (store[key][1][i] - min(store[key][1][i]))/(max(store[key][1][i]) - min(store[key][1][i]))
                        array = resize(array,(array.shape[0],array.shape[1],1))
                        array = resize_with_crop_or_pad(array,size,size)
                        array = transpose(array,(2,0,1))
                        test_images[(k*length)+i,:,:] = array
                        test_labels[(k*length)+i] = loc
                    k += 1
                else:
                    for i in range(len(store[key])):
                        array = (store[key][1][i] - min(store[key][1][i]))/(max(store[key][1][i]) - min(store[key][1][i]))
                        array = resize(array,(array.shape[0],array.shape[1],1))
                        array = resize_with_crop_or_pad(array,size,size)
                        array = transpose(array,(2,0,1))
                        train_images[(j*length)+i,:,:] = array
                        train_labels[(j*length)+i] = loc
                    j += 1
    return train_images, train_labels, test_images, test_labels, names

# takes list of pdbs and converts to projected mrcs
def makeImages(path, pdbinput):
    from biopandas.pdb import PandasPdb
    import subprocess
    import os
    ppdb = PandasPdb().fetch_pdb(pdbinput)
    COMPNDdf = ppdb.df['OTHERS'].loc[ppdb.df['OTHERS']['record_name']  == 'COMPND']
    moleculeline = COMPNDdf[COMPNDdf['entry'].str.contains(" MOLECULE:")]['entry'].iloc[0]
    moleculename = moleculeline[moleculeline.find(':')+1 : moleculeline.find(';')].replace(' ', '')
    moleculename =''.join(e for e in moleculename if e.isalnum())
    ppdb.to_pdb(path=path + pdbinput + '.pdb')
    print(moleculename)

    # lowpass filter pdb to resolution 3
    subprocess.run(['/programs/x86_64-linux/eman/1.9/bin/pdb2mrc', path + pdbinput + '.pdb', path + moleculename + '_' + pdbinput + '.mrc', 'apix=1', 'res=3', 'center'])
    subprocess.run(['/programs/x86_64-linux/relion/3.0.8/bin/relion_project', '--i', path + moleculename + '_' + pdbinput + '.mrc', '--o', path + moleculename + '_' + pdbinput + '_proj', '--nr_uniform', '30'])


    # delete pdb and mrc files
    os.remove(path + pdbinput + '.pdb')
    try:
        os.remove(path + moleculename + '_' + pdbinput + '.mrc')
    except FileNotFoundError:
        pass
    
    try:
        os.remove(path + moleculename + '_' + pdbinput + '_proj.star')
    except FileNotFoundError:
        pass

    mrcname = moleculename + '_' + pdbinput + '_proj.mrcs'
    return moleculename, mrcname
