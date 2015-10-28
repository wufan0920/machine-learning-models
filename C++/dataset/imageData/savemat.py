import scipy.io as sio
import numpy as np
import copy

def save_mat():
    matfn='../IMAGES.mat'
    images=sio.loadmat(matfn)
    data = images['IMAGES']
    rows = len(data)
    cols = len(data[0])
    for fileno in range(10):
        filename = 'image'+str(fileno)
        f = open(filename,'w')
        for row in range(rows):
            content = ''
            for col in range(cols):
                content+=(str(data[row,col,fileno])+' ')
            content+='\n'
            f.write(content)
        f.flush()
        f.close()
    return

if __name__=='__main__':
    save_mat();
