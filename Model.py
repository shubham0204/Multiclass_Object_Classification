
from tensorflow.python.keras import models , optimizers , losses ,activations
from tensorflow.python.keras.layers import *
from PIL import Image
import tensorflow as tf
import time
import os
import numpy as np

class Classifier (object) :

    def __init__( self , number_of_classes ):

        dropout_rate = 0.5
        self.__DIMEN = 32

        input_shape = ( self.__DIMEN**2 , )
        convolution_shape = ( self.__DIMEN , self.__DIMEN , 1 )
        kernel_size = ( 3 , 3 )
        pool_size = ( 2 , 2 )
        strides = 1

        activation_func = activations.relu

        self.__NEURAL_SCHEMA = [

            Reshape( input_shape=input_shape , target_shape=convolution_shape),

            Conv2D( 32, kernel_size=( 4 , 4 ) , strides=strides , activation=activation_func),
            MaxPooling2D(pool_size=pool_size, strides=strides ),

            Conv2D( 64, kernel_size=( 3 , 3 ) , strides=strides , activation=activation_func),
            MaxPooling2D(pool_size=pool_size , strides=strides),

            Flatten(),

            Dense( 100, activation=activation_func) ,
            Dropout(dropout_rate),

            Dense( number_of_classes, activation=tf.nn.softmax )

        ]

        self.__model = tf.keras.Sequential( self.__NEURAL_SCHEMA )

        self.__model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.categorical_crossentropy ,
            metrics=[ 'accuracy' ] ,
        )

    def fit(self, X, Y  , hyperparameters):
        initial_time = time.time()
        self.__model.fit(X, Y,
                         batch_size=hyperparameters['batch_size'],
                         epochs=hyperparameters['epochs'],
                         callbacks=hyperparameters['callbacks'],
                         validation_data=hyperparameters['val_data']
                         )
        final_time = time.time()
        eta = (final_time - initial_time)
        time_unit = 'seconds'
        if eta >= 60:
            eta = eta / 60
            time_unit = 'minutes'
        self.__model.summary()
        print('Elapsed time acquired for {} epoch(s) -> {} {}'.format(hyperparameters['epochs'], eta, time_unit))

    def prepare_images_from_dir( self , dir_path ) :
        images = list()
        images_names = os.listdir( dir_path )
        for imageName in images_names :
            image = Image.open(dir_path + imageName).convert('L')
            resize_image = image.resize((self.__DIMEN, self.__DIMEN))
            array = list()
            for x in range(self.__DIMEN):
                sub_array = list()
                for y in range(self.__DIMEN):
                    sub_array.append(resize_image.load()[x, y])
                array.append(sub_array)
            image_data = np.array(array)
            image = np.array(np.reshape(image_data,(self.__DIMEN, self.__DIMEN, 1)))
            images.append(image)

        return np.array( images )

    def evaluate(self , test_X , test_Y  ) :
        return self.__model.evaluate(test_X, test_Y)

    def predict(self, X  ):
        predictions = self.__model.predict( X  )
        return predictions

    def save_model(self , file_path ):
        self.__model.save(file_path )

    def load_model(self , file_path ):
        self.__model = models.load_model(file_path)







