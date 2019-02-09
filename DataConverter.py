
import numpy as np
import os
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split

dimen = 32

dir_path = input( 'Enter images directory path : ')
output_path = input( 'Enter output directory path : ')

sub_dir_list = os.listdir( dir_path )
images = list()
labels = list()
for i in range( len( sub_dir_list ) ):
    label = i
    image_names = os.listdir( dir_path + sub_dir_list[i] )
    for image_path in image_names:
        path = dir_path + sub_dir_list[i] + "/" + image_path
        image = Image.open( path ).convert( 'L' )
        resize_image = image.resize((dimen, dimen))
        array = list()
        for x in range(dimen):
            sub_array = list()
            for y in range(dimen):
                sub_array.append(resize_image.load()[x, y])
            array.append(sub_array)
        image_data = np.array(array)
        image = np.array(np.reshape(image_data, (dimen, dimen, 1))) / 255
        images.append(image)
        labels.append( label )

x = np.array( images )
y = np.array( keras.utils.to_categorical( np.array( labels) , num_classes=len(sub_dir_list) ) )

train_features , test_features ,train_labels, test_labels = train_test_split( x , y , test_size=0.4 )

np.save( '{}x.npy'.format( output_path )  , train_features )
np.save( '{}y.npy'.format( output_path )  , train_labels )
np.save( '{}test_x.npy'.format( output_path ) , test_features )
np.save( '{}test_y.npy'.format( output_path ) , test_labels )


