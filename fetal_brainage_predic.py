import numpy as np
import nibabel as nib
import warnings
warnings.filterwarnings(action='ignore') 

import os, glob, sys, argparse
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm




parser = argparse.ArgumentParser('   ==========   Fetal Brain age prediciton by Jinwoo Hong   ==========   ')
parser.add_argument('-input', action='store',dest='input',type=str, required=True, help='input file selection filter e.g. "002*"')
parser.add_argument('-output', action='store',dest='output',type=str, required=True, help='simple result output name e.g. 002_brainage.txt')
parser.add_argument('-detail_output', action='store',dest='d_output',type=str, help='[option] detailed result output name e.g. 002_brainage_full.csv')
parser.add_argument('-batch_size', action='store',dest='bsize',default=80, type=int, help='[option] batch_size e.g. 30')
parser.add_argument('-worker', action='store',dest='worker',default=4, type=int, help='[option] number of batch generator worker (CPU thread) e.g. 4')
args = parser.parse_args()

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def make_dic(img_list):
    max_size = [176, 138]
    dic = np.zeros([len(img_list)*4, max_size[1], max_size[0], 3])
    for i in tqdm(range(0, len(img_list)),desc='Making dic'):
        img = np.squeeze(nib.load(img_list[i]).get_data())
        if (np.asarray(img.shape[:2]) > max_size).any():
            dif = np.asarray(img.shape[:2])-max_size
            if dif[0]>0:
                img = img[int(np.ceil((img.shape[0]-max_size[0])/2)):-int((img.shape[0]-max_size[0])/2),:,:]
            if dif[1]>0:
                img = img[:,int(np.ceil((img.shape[1]-max_size[1])/2)):-int((img.shape[1]-max_size[1])/2),:]

        img = np.pad(img,((int(np.ceil((max_size[0]-img.shape[0])/2)),int((max_size[0]-img.shape[0])/2)),
                   (int(np.ceil((max_size[1]-img.shape[1])/2)),int((max_size[1]-img.shape[1])/2)),
                   (0,0)), 'constant')
        # img = (img-np.mean(img))/np.std(img)
        dic[i*4:i*4+4,:,:,0]=np.swapaxes(img[:,:,int(img.shape[-1]/2)-2:int(img.shape[-1]/2)+2],0,2)
        dic[i*4:i*4+4,:,:,1]=dic[i*4:i*4+4,:,:,0]
        dic[i*4:i*4+4,:,:,2]=dic[i*4:i*4+4,:,:,0]
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
        pred = model.predict_generator(datagen.flow(image, batch_size=n_example),steps=n_example, workers = args.worker, verbose=0)
        preds[i]=np.mean(pred)
    return preds

img_list = np.asarray(sorted(glob.glob(args.input)))

test_dic = make_dic(img_list)

model = age_predic_network([138,176,3])
model.load_weights(resource_path('data/best_fold0_rsl.h5'))

p_age2 = tta_prediction(datagen,model, test_dic,20)

print('\n\n\t\t Estimated brain age : \t\t\t '+str(np.mean(p_age2))+'\n\n')

np.savetxt(args.output,np.mean(p_age2)[np.newaxis],fmt="%f")
if args.d_output != None:
    np.savetxt(args.d_output,np.concatenate((img_list[:,np.newaxis],p_age2.reshape(-1,4)),axis=1),fmt="%s",delimiter=',')
    

