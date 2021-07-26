import numpy as np
import nibabel as nib
import warnings
warnings.filterwarnings(action='ignore') 
import pandas as pd
import os, glob, sys, argparse
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

parser = argparse.ArgumentParser('   ==========   Fetal Brain age prediciton by Jinwoo Hong   ==========   ')
parser.add_argument('-input_csv', action='store',dest='input',type=str, required=True, help='input file selection filter e.g. "002*"')
parser.add_argument('-output', action='store',dest='output',type=str, required=True, help='output csv name')
parser.add_argument('-weight', action='store',dest='weight',type=str, required=True, help='name of trained weight file')
parser.add_argument('-n_slice',action='store',dest='num_slice',default=4,type=int, required=True, help='Number of training slice from a volume')
parser.add_argument('-gpu',action='store',dest='num_gpu',default='0', type=str, help='GPU selection')
parser.add_argument('-batch_size', action='store',dest='bsize',default=80, type=int, help='[option] batch_size e.g. 30')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.num_gpu

def _get_mode(predic_age, bin_range=0.2):
    bin_list=np.arange(np.floor(np.min(predic_age))-5,np.ceil(np.max(predic_age))+5,bin_range)
    bin_count=np.zeros([len(bin_list),])
    for i in range(0,len(predic_age)):
        bin_count[np.bitwise_and((predic_age[i]-bin_list)<=(bin_range/2),(predic_age[i]-bin_list)>=(-bin_range/2))] = bin_count[np.bitwise_and((predic_age[i]-bin_list)<=(bin_range/2),(predic_age[i]-bin_list)>=(-bin_range/2))]+1
    if np.sum(bin_count==max(bin_count))>1:
        pred_sub_argmax = 3*np.median(predic_age) - 2*np.mean(predic_age)
    else:
        j=np.where(bin_count==np.max(bin_count))[0][0]
        f=bin_count[j-1:j+2]
        L = bin_list[j]-bin_range/2
        pred_sub_argmax = L + bin_range*((f[1]-f[0])/((2*f[1]) - f[0] - f[2]))

    return pred_sub_argmax

def crop_pad_ND(img, target_shape):
    import operator, numpy as np
    if (img.shape > np.array(target_shape)).any():
        target_shape2 = np.min([target_shape, img.shape],axis=0)
        start = tuple(map(lambda a, da: a//2-da//2, img.shape, target_shape2))
        end = tuple(map(operator.add, start, target_shape2))
        slices = tuple(map(slice, start, end))
        img = img[tuple(slices)]
    offset = tuple(map(lambda a, da: a//2-da//2, target_shape, img.shape))
    slices = [slice(offset[dim], offset[dim] + img.shape[dim]) for dim in range(img.ndim)]
    result = np.zeros(target_shape)
    result[tuple(slices)] = img
    return result


def make_dic(img_list, num_slice, slice_mode=0, desc=''):
    max_size = [176, 138, 1]
    if slice_mode:
        dic = np.zeros([len(img_list), max_size[1], max_size[0], num_slice],dtype=np.float16)
    else:
        dic = np.zeros([len(img_list)*num_slice, max_size[1], max_size[0], 1],dtype=np.float16)
    for i in tqdm(range(0, len(img_list)),desc=desc):
        img = np.squeeze(nib.load(img_list[i]).get_fdata())
        img = crop_pad_ND(img, np.max(np.vstack((max_size, img.shape)),axis=0))
        # img = (img-np.mean(img))/np.std(img)
        if slice_mode:
            dic[i,:,:,:]=np.swapaxes(img[:,:,int(img.shape[-1]/2-1-np.int(num_slice/2)):int(img.shape[-1]/2+np.int(num_slice/2))],0,1)
        else:
            dic[i*num_slice:i*num_slice+num_slice,:,:,0]=np.swapaxes(img[:,:,int(img.shape[-1]/2-1-np.int(num_slice/2)):int(img.shape[-1]/2+np.int(num_slice/2))],0,2)
            # dic[i*num_slice:i*num_slice+num_slice,:,:,1]=dic[i*num_slice:i*num_slice+num_slice,:,:,0]
            # dic[i*num_slice:i*num_slice+num_slice,:,:,2]=dic[i*num_slice:i*num_slice+num_slice,:,:,0]
    return dic

def age_predic_network(img_shape):
    from keras.layers import Dense, Input, concatenate, Dropout, Flatten
    from keras.models import Model
    from keras.optimizers import Adam
    from keras_applications.resnet_v2 import ResNet101V2
    import keras.backend as K
    import keras
    model = ResNet101V2(input_shape=img_shape,include_top=False, weights=None, pooling='avg',backend=keras.backend,layers=keras.layers,models=keras.models, utils=keras.utils)
    o = Dropout(0.3)(model.layers[-1].output)
    o = Dense(1,activation='linear')(o)
    model = Model(model.layers[0].output, o)
    return model


datagen = ImageDataGenerator(
    rotation_range=360,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.5,1],
    vertical_flip=True,
    horizontal_flip=True)


def tta_prediction(datagen, model, dic, n_example):
    preds=np.zeros([len(dic),])
    for i in tqdm(range(len(dic)),desc='TTA prediction'):
        image = np.expand_dims(dic[i],0)
        pred = model.predict_generator(datagen.flow(image, batch_size=n_example),steps=n_example, workers = 4, verbose=0)
        preds[i]=np.mean(pred)
    return preds


info = pd.read_csv(args.input)
b_test_ID = []
for tt in range(len(info['ID'])):
    b_test_ID = np.concatenate((b_test_ID,np.tile(info['ID'][tt],(args.num_slice,))),axis=0)

test_dic = make_dic(info['MR'].values, args.num_slice)

model = age_predic_network([138,176,3])
model.load_weights(args.weight)

p_age2 = tta_prediction(datagen,model, test_dic,20)

id_uniq = np.unique(info['ID'])

predic = np.zeros([len(id_uniq),])
for i in range(0,len(id_uniq)):
    search = np.zeros(len(b_test_ID,))
    for i2 in range(len(b_test_ID)):
        search[i2] = b_test_ID[i2].find(id_uniq[i])
    predic[i]=_get_mode(p_age2[search==0])

result = pd.DataFrame(np.concatenate((id_uniq[:,np.newaxis],predic[:,np.newaxis]),axis=1),columns=['ID','GW'])
result.to_csv(args.output, index=False)

