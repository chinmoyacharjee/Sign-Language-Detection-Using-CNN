
#importing all libraies

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras import initializers
from keras import optimizers

def create_model():
    classifier = Sequential()
    
    # Adding a first convolutional layer
    classifier.add(Conv2D(16, (5, 5), input_shape = (80, 80, 3), activation = 'relu', kernel_initializer=initializers.random_normal(stddev=0.04,mean = 0.00), bias_initializer = initializers.Constant(value=0.2)))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (5, 5), activation = 'relu', kernel_initializer=initializers.random_normal(stddev=0.04,mean = 0.00), bias_initializer = initializers.Constant(value=0.2)))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding a third convolutional layer
    classifier.add(Conv2D(48, (4, 4), activation = 'relu', kernel_initializer=initializers.random_normal(stddev=0.04,mean = 0.00), bias_initializer = initializers.Constant(value=0.2)))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Flattening
    classifier.add(Flatten())
    
    #Full connection
    classifier.add(Dense(512, activation = 'relu', kernel_initializer=initializers.random_normal(stddev=0.02,mean = 0.00), bias_initializer = initializers.Constant(value=0.1)))
    
    # output layer    
    classifier.add(Dense(11, activation = 'softmax', kernel_initializer=initializers.random_normal(stddev=0.02,mean = 0.00), bias_initializer = initializers.Constant(value=0.1)))    
    
    return classifier

classifier = create_model()

#classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#using optimizer sgd
sgd = optimizers.SGD(lr=1e-2)
classifier.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 3,
                                   zoom_range = 0.1,
                                   fill_mode='nearest',
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 3,
                                   zoom_range = 0.1,
                                   fill_mode='nearest',
                                   horizontal_flip = True)


training_set = train_datagen.flow_from_directory('DigitDataBin2/train_set',
                                                 target_size = (80, 80),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('DigitDataBin2/test_set',
                                            target_size = (80, 80),
                                            batch_size = 32,
                                            class_mode = 'categorical')

test_set.class_indices
classifier.fit_generator(training_set,
                         steps_per_epoch = 22996//32 ,
                         validation_data=test_set,
                         epochs = 5,
                         validation_steps = 7579//32)

classifier.save('digit_model_0_10_OP3.h5') 

