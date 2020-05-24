
# function to sort projected images into dictionary as arrays
def GetImageSet(mrc,name,store):
    from mrcfile import open
    with open(mrc,'r') as m:
        data = []
        for i in range(m.data.shape[0]):
            data.append(m.data[i,:,:])
    store[name] = data