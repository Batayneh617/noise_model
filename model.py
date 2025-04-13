from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, LeakyReLU, SpatialDropout2D
from keras.regularizers import l2

def get_conv_model(input_shape):
    model = Sequential()

    # First conv block
    model.add(Conv2D(32, (7, 7), activation='tanh', padding='same', 
                     input_shape=input_shape, kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(64, (5, 5), activation='tanh', padding='same', kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.3))

    # Second conv block
    model.add(Conv2D(128, (3, 3), activation='tanh', padding='same', kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(128, (3, 3), activation='tanh', padding='same', kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.3))



    # Global Average Pooling instead of Flatten
    model.add(GlobalAveragePooling2D())

    # Fully Connected layers
    model.add(Dense(64, activation='tanh', kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='tanh', kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(3, activation='softmax'))  # 3 classes: wind, rain, mic noise

    return model