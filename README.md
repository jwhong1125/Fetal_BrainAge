# Fetal_BrainAge
Optimal Method for Fetal Brain Age Prediction  using Multiplanar Slices from Structural MRI

## Abstract
Accurate prediction of fetal brain age using magnetic resonance imaging (MRI) may contribute to identifying brain abnormalities and the risk of adverse developmental outcomes. This study proposes a method to predict fetal brain age using MRIs from 220 healthy fetuses between 15.9 and 38.7 weeks of gestational age (GA). We built a 2D single-channel convolutional neural network (CNN) with multiplanar MRI slices in different orthogonal planes without correction for interslice motion. In each fetus, multiple age predictions from different slices were generated, and its brain age was obtained by the mode that finds the most frequent value among the multiple predictions from 2D single-channel CNN. We yielded a mean absolute error (MAE) of 0.125 weeks (0.875 days), R2 = 0.999, and p < 0.001 between GA and brain age across fetuses. The use of multiplanar images achieved significantly lower prediction error and variance compared with using a single slice and a single MRI stack of slices. Our 2D single-channel CNN with multiplanar slices obtained significantly lower MAE than 2D multi-channel and 3D CNNs. Saliency maps from our method indicate that anatomical information describing the cortex and ventricles were primary contributors to brain age prediction. When applying the proposed method to external MRIs from 21 healthy fetuses, the use of multiplanar slices and the mode also significantly reduced prediction error and variance (MAE of 0.508 weeks, R2 = 0.987, and p < 0.001). These results demonstrate that our method with multiplanar slices accurately predicts fetal brain age without the need for increased dimensionality or complex preprocessing steps.

## Requirements
```
pip3 install -r requirements.txt
```

## Training
```
python3 fetal_brainage_train.py -input_csv sample.csv
```
### Required arguments
* -input_csv: input csv table for data information (see sample.csv)
### Optional arguments
* -n_slice: number of slice from each MRI stack for training, default=4
* -batch_size: number of batch, default=80
* -slice_mode: 0: multi-single-channel training, 1: multi-channel training, default=0
* -tta: number of tta, default=20
* -f: number of fold for training, default=10
* -fs: start fold number, default=0
* -fe: end fold number, default=10
* -d_huber: delta value of huber loss, default=1.0
* -gpu: number of gpu to use, default=0
* -rl: result output path, default=./
* -wl: weight output path, default=./
* -hl: training history output path, default=./

## Test
```
python3 fetal_brainage_test.py -input_csv sample.csv -output predic_result.csv -weight trained_weight.h5
```
### Required arguments
* -input_csv: input csv table for data information (see sample.csv)
* -output: output csv name
* -weight: name of trained weight file
### Optional arguments
* -n_slice: number of slice from each MRI stack for training, default=4
* -gpu: number of gpu to use, default=0
* -batch_size: number of batch, default=80
