
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import regularizers
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf


# dimensions of our images.
img_width, img_height = 200, 200

top_model_weights_path = r'bottleneck_fc_model_randlayermy.h5'
train_data_dir = r'D:\aesta\tester'

nb_train_samples = 7834
nb_validation_samples = 1742
epochs = 15
batch_size = 20
batch_size1=20


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)
   
    base_model = applications.InceptionV3(include_top=False, weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer(name='mixed2').output)
    
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size1,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(generator)


    np.save(open('timepass.npy', 'wb'),bottleneck_features_train)
    train_data = np.load(open('timepass.npy', 'rb'))

    print(np.shape(train_data))



    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=train_data.shape[1:]))
    model.add(Activation('relu'))
  

    model.add(Conv2D(32, (3, 3), ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3),kernel_regularizer=regularizers.l1(0.01)))
    model.add(Activation('relu'))
    

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model = load_model('inceptionrandlayerdataaugmentmodel.h5')


    
    sumo=0
    for i in range(0,10):
        
        x=train_data[i]
        x = np.expand_dims(x, axis=0)
        preds = model.predict_classes(x)
        probs = model.predict_proba(x)
        print("%s   %s" % (probs, preds))
        sumo=sumo+1
          
    print(sumo)
    print('done')


save_bottlebeck_features()

