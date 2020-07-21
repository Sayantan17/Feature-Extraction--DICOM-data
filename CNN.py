## A general glimpse!!!!!!!

## Load the data
import dicom
import os
import numpy

Path = "./dir_with_dicom_series/"
files_dcm = []  
for dir_name, subdirList, fileList in os.walk(Path):
    for filename in fileList:
        if ".dcm" in filename.lower():  
            lstFilesDCM.append(os.path.join(dir_name,filename))
            
# Get ref file
RefDs = dicom.read_file(files_dcm[0])

## say along the saggital planes
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(files_dcm))

# Load spacing values (in mm)
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

# The array is sized based on 'ConstPixelDims'
ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

# loop through all the DICOM files
for filenameDCM in files_dcm:
    # read the file
    ds = dicom.read_file(filenameDCM)
    # store the raw image data
    ArrayDicom[:, :, files_dcm.index(filenameDCM)] = ds.pixel_array

### Array created for a folder having the folder's data

## Load the ground truths with labels

## ROI creation TARGET- 28x28x10 the 128x128x50 size of one dcm file
    
#$# Can be done with Slicer 3d--- fopr which we have got a numpy array too! Can be done in another script
######\\\\\    
import os
import dicom
import numpy
import pandas as pd

from __main__ import vtk, qt, ctk, slicer
import sitkUtils

 print("Cropped CT volumes")

 inputVolume  = ./mydata_folder.dcm"  
 croppedImage_fol = "./myCroppedVolume.dcm"

 [success, inputVolume] = slicer.util.loadVolume(inputVolume, returnNode=True)          
 inputImage = sitkUtils.PullVolumeFromSlicer(inputVolume.GetID())

 cropper = sitkUtils.sitk.CropImageFilter()
 croppingBounds = [[178, 210, 67],[227, 195, 34] #input the dimensions of the region.
 croppedImage = cropper.Execute(inputImage, croppingBounds[0], croppingBounds[1]) 
 croppedNode = sitkUtils.PushVolumeToSlicer(croppedImage, None,  inputVolume.GetName() , 
 'vtkMRMLScalarVolumeNode' )

 properties = {}
 properties["fileType"] = ".dcm"
 slicer.util.saveNode(croppedNode,  croppedImage_fol, properties)

 exit()
    
## Convert all the cropped data into numpy!
 
 ################### Another format


### Analysis part
data_dir='...///'
pat=os.listdir.(data_dir)
labels=pd.read_csv('',index_col=0)
labels_df.head()
for pat in pats[:1]:
    label=labels_df.get_value(pat,'')
    path=data_dir+pat
    slices=[dicom.read_file(path+'/'+s) for s in os.listdir(path)]
    slices.sort(key= lambda x: int(x.   ImagePositionPatient[2]))
    print(len(slices),slices[0].pixel_array.shape)
#### Training part-///// 3D CNNs
 
import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D)
from keras.layers import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split


## Plotting the data
def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()
    
## Training/testing history
def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))    


## Load the data that was earlier in the form of numpy array
## Training and testing can be created from that
## As we have the corresponding labels----- in the numpy array! 


## Here we go the data loading step
data = np.load('data.npy') ############  All the data
# could probably work on these values for the training/validation set to better train the CNN for other data sets
training_set = data[:100]
validation_set = data[-100:]           
            
def main():

    parser = argparse.ArgumentParser(
        description='SIMPLE PARSER FOR THE CLASSIFICATION')
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--nclass', type=int, default=04)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--slicing', type=int, default=8)
    args = parser.parse_args()

    img_rows, img_cols, slicing = 32, 32, args.slicing
    channel = 1 ###############graysccale format
    td_npz = 'dataset_{}_{}_{}.npz'.format(
        args.nclass, args.slicing, args.skip)

    nb_classes = args.nclass
#########
    loadeddata = np.load(td_npz)
    X, Y = loadeddata["X"], loadeddata["Y"]
########
######
    X = x.reshape((x.shape[0], img_rows, img_cols, slicing, channel))
    Y = np_utils.to_categorical(y, nb_classes)

    X = X.astype('float32')
    np.savez(td_npz, X=X, Y=Y)
    print('Saved dataset to dataset.npz.')
    print('X_shape:{}\nY_shape:{}'.format(X.shape, Y.shape))

 
 
##########----------------------------------------------------------------------------------------------

# Model Architecture --- just directly feeding in the input ROIs

    model=sequential()
    model.add(Conv3D(32, (3,3,3), activation='LeakyRelu', input_shape=(6, 36, 36, 8, 1)))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='valid', strides=(2, 2, 2)))
    model.add(Conv3D(48, (3,3,3), activation='LeakyRelu'))
    model.add(Conv3D(128, (3,3,3), activation='LeakyRelu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1)))
    model.add(Conv3D(256, (2,2,2), activation='LeakyRelu'))
    # FC Layer
    model.add(Flatten())
    model.add(Dense(128, activation='LeakyRelu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='LeakyRelu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss=categorical_crossentropy, optimizer='Adam', metrics=['accuracy'])
    model.summary()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1)
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=args.batch,
                        epochs=args.epoch, verbose=1, shuffle=True)
    model.evaluate(X_test, Y_test, verbose=0)


    ##### MAIN PART
    if __name__=='__main__'
         main()
