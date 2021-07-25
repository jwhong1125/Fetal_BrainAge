import numpy as np
import nibabel as nib
import os, glob, sys, time, pickle
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
import argparse
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser('   ==========   Fetal brain age prediction, made by Jinwoo Hong 2020.09.20 ver.1)   ==========   ')
parser.add_argument('-input_csv',action='store',dest='input_csv',type=str, required=True, help='input csv table')
parser.add_argument('-batch_size',action='store',default=80,dest='num_batch',type=int, help='Number of batch')
parser.add_argument('-n_slice',action='store',dest='num_slice',default=1,type=int, required=True, help='Number of training slice from a volume')
parser.add_argument('-slice_mode',action='store',dest='slice_mode',default=0,type=int, required=True, help='0: multi-slice training, 1: multi-channel training')
parser.add_argument('-tta',action='store',dest='num_tta',default=20, type=int, help='Number of tta')
parser.add_argument('-f',action='store',dest='num_fold',default=10, type=int, help='number of fold for training')
parser.add_argument('-fs', action='store',dest='start_fold', default=0, type=int, help='start fold number')
parser.add_argument('-fe', action='store',dest='end_fold', default=10, type=int, help='end fold number')
parser.add_argument('-d_huber', action='store',dest='delta_huber', default=1.0, type=float, help='delta value of huber loss')
parser.add_argument('-gpu',action='store',dest='num_gpu',default='0', type=str, help='GPU selection')
parser.add_argument('-rl', '--result_save_locaiton', action='store',default='./', dest='result_loc', required=True, type=str, help='Output folder name, default: ./')
parser.add_argument('-wl', '--weight_save_location', action='store',default='./', dest='weight_loc', required=True, type=str, help='Output folder name, default: ./')
parser.add_argument('-hl', '--history_save_location', action='store',default='./', dest='hist_loc', required=True, type=str, help='Output folder name, default: ./')
args = parser.parse_args()

result_loc=args.result_loc
weight_loc=args.weight_loc
hist_loc=args.hist_loc

if os.path.exists(result_loc)==False:
    os.makedirs(result_loc,exist_ok=True)
if os.path.exists(weight_loc)==False:
    os.makedirs(weight_loc, exist_ok=True)
if os.path.exists(hist_loc)==False:
    os.makedirs(hist_loc, exist_ok=True)

print('\n\n')
print('\t\t Input csv: \t\t\t\t\t\t'+os.path.realpath(args.input_csv))
print('\t\t Prediction result save location: \t\t\t'+os.path.realpath(result_loc))
print('\t\t Prediction weights save location: \t\t\t'+os.path.realpath(weight_loc))
print('\t\t Prediction history save location: \t\t\t'+os.path.realpath(hist_loc))
print('\t\t Total fold: \t\t\t\t\t\t'+str(args.num_fold))
print('\t\t Start fold: \t\t\t\t\t\t'+str(args.start_fold))
print('\t\t End   fold: \t\t\t\t\t\t'+str(args.end_fold))
print('\t\t number training slice: \t\t\t\t'+str(args.num_slice))
print('\t\t Slice mode: \t\t\t\t'+str(args.slice_mode))
print('\t\t TTA times: \t\t\t\t\t\t'+str(args.num_tta))
print('\t\t batch_size: \t\t\t\t\t\t'+str(args.num_batch))
print('\t\t delta of Huber loss: \t\t\t\t\t'+str(args.delta_huber))
print('\t\t GPU number: \t\t\t\t\t\t'+str(args.num_gpu))
print('\n\n')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.num_gpu

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
set_session(tf.Session(config=config))

batch_size = args.num_batch
num_slice=args.num_slice

def huber_loss(y_true, y_pred, delta=args.delta_huber):
    error = y_pred - y_true
    abs_error = K.abs(error)
    quadratic = K.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * K.square(quadratic) + delta * linear

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
    model.compile(optimizer=Adam(lr=0.1,decay=0.001), loss=huber_loss, metrics=['mae'])
    return model

info = pd.read_csv(args.input_csv)
info.insert(5,'GW_round', info.GW.astype(float).round(0).astype(int))
skf = StratifiedKFold(n_splits=args.num_fold, random_state=1,shuffle=True)
fold_info = list(skf.split(info, info.GW_round.values))

datagen = ImageDataGenerator(
    rotation_range=360,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.5,1],
    vertical_flip=True,
    horizontal_flip=True)


def tta_prediction(datagen, model, dic, n_example):
    preds=np.zeros([len(dic),])
    for i in range(len(dic)):
        image = np.expand_dims(dic[i],0)
        pred = model.predict_generator(datagen.flow(image, batch_size=n_example),workers=4,steps=n_example, verbose=0)
        preds[i]=np.mean(pred)
    return preds

for ii in range(args.start_fold,args.end_fold):
    tr2 = fold_info[ii][0]
    te = fold_info[ii][1]
    in_fold_info = list(skf.split(info.iloc[tr2],info.iloc[tr2].GW_round.values))[0]
    tr = tr2[in_fold_info[0]]
    va = tr2[in_fold_info[1]]

    for i in info.iloc[tr].GW_round.unique():
        if i == info.iloc[tr].GW_round.unique()[0]:
            info_t = info.iloc[tr][info['GW_round'].iloc[tr]==i].sample(info.iloc[tr].GW_round.value_counts().values[0]+int(info.iloc[tr].GW_round.value_counts().values[0]/info.iloc[tr][info['GW_round'].iloc[tr]==i].shape[0]),replace=True)
            continue
        info_t = pd.concat([info_t, info.iloc[tr][info['GW_round'].iloc[tr]==i].sample(info.iloc[tr].GW_round.value_counts().values[0]+int(info.iloc[tr].GW_round.value_counts().values[0]/info.iloc[tr][info['GW_round'].iloc[tr]==i].shape[0]),replace=True)] ,axis=0, ignore_index=1)

    for i in info.iloc[va].GW_round.unique():
        if i == info.iloc[va].GW_round.unique()[0]:
            info_v = info.iloc[va][info['GW_round'].iloc[va]==i].sample(info.iloc[va].GW_round.value_counts().values[0]+int(info.iloc[va].GW_round.value_counts().values[0]/info.iloc[va][info['GW_round'].iloc[va]==i].shape[0]),replace=True)
            continue
        info_v = pd.concat([info_v, info.iloc[va][info['GW_round'].iloc[va]==i].sample(info.iloc[va].GW_round.value_counts().values[0]+int(info.iloc[va].GW_round.value_counts().values[0]/info.iloc[va][info['GW_round'].iloc[va]==i].shape[0]),replace=True)] ,axis=0, ignore_index=1)

    if args.slice_mode:
        train_dic = make_dic(info_t.MR.values, num_slice, slice_mode=1, desc='make train dic')
        val_dic = make_dic(info_v.MR.values, num_slice, slice_mode=1, desc='make val dic')
        test_dic = make_dic(info.iloc[te].MR.values, num_slice, slice_mode=1, desc='make test dic')

        b_train_GW = info_t.GW.values
        b_val_GW = info_v.GW.values
        b_test_GW = info.iloc[te].GW.values
        b_test_ID = info.iloc[te].ID.values

        model = age_predic_network([138,176,num_slice])
    else:
        train_dic = make_dic(info_t.MR.values, num_slice, slice_mode=0, desc='make train dic')
        val_dic = make_dic(info_v.MR.values, num_slice, slice_mode=0, desc='make val dic')
        test_dic = make_dic(info.iloc[te].MR.values, num_slice, slice_mode=0, desc='make test dic')

        train_GW = info_t.GW.values
        b_train_GW = np.zeros([len(train_GW)*num_slice,])
        for tt in range(len(train_GW)):
            b_train_GW[tt*num_slice:tt*num_slice+num_slice,]=np.tile(train_GW[tt],(num_slice,))
        val_GW = info_v.GW.values
        b_val_GW = np.zeros([len(val_GW)*num_slice,])
        val_ID = info_v.ID.values
        b_val_ID = []
        for tt in range(len(val_GW)):
            b_val_ID = np.concatenate((b_val_ID,np.tile(val_ID[tt],(num_slice,))),axis=0)

        for tt in range(len(val_GW)):
            b_val_GW[tt*num_slice:tt*num_slice+num_slice,]=np.tile(val_GW[tt],(num_slice,))
        test_GW = info.iloc[te].GW.values
        b_test_GW = np.zeros([len(test_GW)*num_slice,])
        for tt in range(len(test_GW)):
            b_test_GW[tt*num_slice:tt*num_slice+num_slice,]=np.tile(test_GW[tt],(num_slice,))
        test_ID = info.iloc[te].ID.values
        b_test_ID = []
        for tt in range(len(test_GW)):
            b_test_ID = np.concatenate((b_test_ID,np.tile(test_ID[tt],(num_slice,))),axis=0)

        model = age_predic_network([138,176,1])

    callbacks = [EarlyStopping(monitor='val_mean_absolute_error', patience=150, verbose=1, mode='min'),
                ModelCheckpoint(filepath=weight_loc+'/best_fold'+str(ii)+'_rsl.h5', monitor='val_mean_absolute_error', save_best_only=True, mode='min', save_weights_only=True, verbose=0)]
    histo = model.fit_generator(datagen.flow(train_dic,b_train_GW,batch_size=batch_size,shuffle=True),steps_per_epoch=len(train_dic)/batch_size,epochs=10000, validation_data=datagen.flow(val_dic, b_val_GW, batch_size=batch_size,shuffle=True),validation_steps=len(val_dic),workers=4,callbacks=callbacks)
    with open(hist_loc+'/history_fold'+str(ii)+'_rsl.pkl', 'wb') as file_pi:
            pickle.dump(histo.history, file_pi)
    model.load_weights(weight_loc+'/best_fold'+str(ii)+'_rsl.h5')
    #p_age = model.predict(test_dic,batch_size=batch_size)
    p_age2 = tta_prediction(datagen,model, test_dic,20)
    #np.savetxt(result_loc+'/fold'+str(ii)+'_rsl.txt',np.concatenate((b_test_ID[:,np.newaxis],b_test_GW[:,np.newaxis],p_age),axis=1),fmt="%s")
    np.savetxt(result_loc+'/fold'+str(ii)+'_aug_rsl.txt',np.concatenate((b_test_ID[:,np.newaxis],b_test_GW[:,np.newaxis],p_age2[:,np.newaxis]),axis=1),fmt="%s")

    del model, histo, p_age, p_age2

    K.clear_session()
    tf.reset_default_graph()

