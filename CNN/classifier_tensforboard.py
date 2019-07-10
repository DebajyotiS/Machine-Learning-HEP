#Prompt photon classifier using Conv Net
#Original Author: Shamik Ghosh, SINP

#Code port for Python 3.x. Check local keras and tensorflow version with keras.__version__ ,
# tensorflow.__version__ before using code.

#importing libraries, modules, classes

import tensorflow as tf
import numpy as np
import graphviz
import matplotlib.pyplot as plt
import keras
from keras import optimizers
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from keras import regularizers
from sklearn.metrics import roc_curve,auc


#Arrays from which the model will select the appropriate value and train itself.
#call "tensorboard --logdir=logs/" (without quotations, from the directory containing the logs/ folder.)
#The tensorboard api will show all the models. Chose the model with desirable loss and val_loss.
#The code generates the logs/ folder

# dense_layers = [0,1,2]
# layer_sizes = [80]
# conv_layers = [0,1,2,3]

dense_layers = [2]
layer_sizes = [80]
conv_layers = [3]

slow_adam = optimizers.Adam(lr= 0.0001)


#The training and Validation data is in output0. <-- Obtained from Geant4. Reshaping the data
data_set = np.loadtxt("output0.csv")
(x_train, y_train) = (data_set[:,0:81],data_set[:,81])
X_Net = x_train.reshape(x_train.shape[0],9,9,1)
Y_Net = to_categorical(y_train,2)


#This is a simple book keeping counter. For higher number of models being trained, it is desirable
#to have a counter showing which model is being trained.
model_number = 0


#This nested loop will go through all the values in the arrays defined above and train the generated model.
#Takes ~10 min on an i5 8th gen, 4 GB Ram, 1 TB 5400 HDD

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            model_number+=1

            #Naming the model
            NAME = f'Prmpt_phtns_clsfr-{conv_layer+1}-conv-{layer_size}-nodes-{dense_layer}-dense'
            #Creating the logs folder and appending it with the model
            tboard = TensorBoard(log_dir='logs/{}'.format(NAME))


            #model definition starts here
            model = Sequential()

            model.add(Conv2D(layer_size,(3,3), strides = 1 , padding="same", activation="relu", input_shape = (9,9,1), kernel_regularizer=regularizers.l2(0.01)))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size,(3,3), strides = 1, padding='same',activation='relu'))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten())

            for l in range(dense_layer):
                model.add(Dense(layer_size, activation='relu'))
                # model.add(Dropout(0.2))

            model.add(Dense(Y_Net.shape[1], activation='softmax'))

            model.compile(loss='binary_crossentropy', optimizer = slow_adam, metrics = ['binary_accuracy'])
#             model.summary()
            print(f'Training Model Number:{model_number}')
            model.fit(X_Net,Y_Net, batch_size=200, epochs=20, shuffle = True, validation_split=0.2, callbacks=[tboard])
