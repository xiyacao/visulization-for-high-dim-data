import numpy as np
import sklearn

def load_shoe_data(sub, data_path='/home/xiya/ROAR/data/ShoesData/saveDB/'):
    subID = 'Sbj%03d'%sub
    datapath = data_path + subID + '.pickle'
    dataset = np.load(datapath)
    dataset = dateset[0]
    shoe_r= dataset['shoe_r']
    label_r = dataset['y_r']
    return shoe_r, label_r
