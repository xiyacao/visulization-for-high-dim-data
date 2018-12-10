import numpy as np
import sklearn

def load_shoe_data(sub, side='r', data_path='/home/xiya/ROAR/data/ShoesData/saveDB/'):
    subID = 'Sbj%03d'%sub
    datapath = data_path + subID + '/' + subID + '.pickle'
    dataset = np.load(datapath)
    dataset = dataset[0]
    if side is 'r':
        shoe = dataset['shoe_r']
        label = dataset['ymat_r']
    else:
    	shoe = dataset['shoe_l']
    	label = dataset['ymat_l']
    length = min(shoe.shape[0], label.shape[0])
    shoe = shoe[:length, :]
    label = label[:length]
    # import IPython
    # IPython.embed()
    return shoe, label

def sample_shoe_data(sub, samples=1000, side='r', data_path='/home/xiya/ROAR/data/ShoesData/saveDB/'):
	shoe, label = load_shoe_data(sub, side, data_path)
	length = len(label)
	indexes = np.random.choice(length, samples)
	sampled_shoe = shoe[indexes, :]
	sampled_label = label[indexes]
	return sampled_shoe, sampled_label


def scale_data(data):
	# data in the form: data_points * channels
	channels = data.shape[1]
	for i in range(channels):
		channel_mean = np.mean(data[:,i])
		channel_scale = max(data[:,i]) - min(data[:,i])
		data[:, i] = (data[:, i] - channel_mean) / channel_scale
	return data

def scale_channel(data):
	# data in the form: data_points * channels
	channel_mean = np.mean(data.flatten())
	channel_scale = max(data.flatten()) - min(data.flatten())
	data = (data - channel_mean) / channel_scale
	return data


def sample_back_shoe_data(sub, samples=10000, sample_back=50, side='r', data_path='/home/xiya/ROAR/data/ShoesData/saveDB/'):
	shoe, label = load_shoe_data(sub, side, data_path)
	length = len(label)
	channels = shoe.shape[1]
	indexes = np.random.choice(length-sample_back, samples) + sample_back
	sampled_shoe = np.zeros((samples, sample_back, channels))
	for i in range(samples):
		sampled_shoe[i,:,:] = shoe[indexes[i]-sample_back:indexes[i], :]
	for i in range(channels):
		sampled_shoe[:,:,i] = scale_channel(sampled_shoe[:,:,i])
	sampled_shoe = sampled_shoe.reshape((samples, sample_back*channels))
	sampled_label = label[indexes]
	#import IPython
	#IPython.embed()
	return sampled_shoe, sampled_label


