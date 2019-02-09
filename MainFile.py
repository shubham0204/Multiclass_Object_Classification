
from tensorflow.python.keras.callbacks import TensorBoard
from Model import Classifier
import numpy as np
import time

data_dimension = 32

X = np.load( 'processed_data/x.npy'.format( data_dimension ))
Y = np.load( 'processed_data/y.npy'.format( data_dimension ))
test_X = np.load( 'processed_data/test_x.npy'.format( data_dimension ))
test_Y = np.load( 'processed_data/test_y.npy'.format( data_dimension ))

print( X.shape )
print( Y.shape )
print( test_X.shape )
print( test_Y.shape )

X = X.reshape( ( X.shape[0] , data_dimension**2  ) ).astype( np.float32 )
test_X = test_X.reshape( ( test_X.shape[0] , data_dimension**2 ) ).astype( np.float32 )

classifier = Classifier( number_of_classes=8 )
classifier.load_model( 'models/model.h5')

parameters = {
    'batch_size' : 250 ,
    'epochs' : 10 ,
    'callbacks' : None ,
    'val_data' : None
}

classifier.fit( X , Y  , hyperparameters=parameters )
classifier.save_model( 'models/model.h5')

loss , accuracy = classifier.evaluate( test_X , test_Y )
print( "Loss of {}".format( loss ) , "Accuracy of {} %".format( accuracy * 100 ) )

sample_X = classifier.prepare_images_from_dir( 'random_images/' )
sample_X = sample_X.reshape( ( sample_X.shape[0] , data_dimension**2 ) ).astype( np.float32 )
print( classifier.predict( sample_X ).argmax( 1 ) )



