from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout

from metrics import keras_iou

def model():
    input_conv = Input((128, 128, 3))

    s = Lambda(lambda x: x / 255) (input_conv)

    # smaller
    conv_1 = Conv2D(16, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(s)
    dropout_1 = Dropout(0.1)(conv_1)
    conv_2 = Conv2D(16, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(dropout_1)
    pooling_1 = MaxPooling2D((2,2))(conv_2)

    conv_3 = Conv2D(32, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(pooling_1)
    dropout_2 = Dropout(0.1)(conv_3)    
    conv_4 = Conv2D(32, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(dropout_2)
    pooling_2 = MaxPooling2D((2,2))(conv_4)

    conv_5 = Conv2D(64, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(pooling_2)
    dropout_3 = Dropout(0.2)(conv_5)    
    conv_6 = Conv2D(64, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(dropout_3)
    pooling_3 = MaxPooling2D((2,2))(conv_6)

    conv_7 = Conv2D(128, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(pooling_3)
    dropout_4 = Dropout(0.2)(conv_7)    
    conv_8 = Conv2D(128, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(dropout_4)
    pooling_4 = MaxPooling2D((2,2))(conv_8)

    # smallest
    conv_9 = Conv2D(256, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(pooling_4)
    dropout_5 = Dropout(0.3)(conv_9)    
    conv_10 = Conv2D(256, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(dropout_5)

    # larger
    conv_10 = Conv2DTranspose(128, (2,2), kernel_initializer='he_normal', strides=(2, 2), padding='same')(conv_10)
    concat_1 = concatenate([conv_10, conv_8])
    conv_11 = Conv2D(128, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(concat_1)
    dropout_6 = Dropout(0.2)(conv_11)    
    conv_12 = Conv2D(128, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(dropout_6)

    conv_13 = Conv2DTranspose(64, (2,2), kernel_initializer='he_normal', strides=(2, 2), padding='same')(conv_12)
    concat_2 = concatenate([conv_13, conv_6])
    conv_14 = Conv2D(64, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(concat_2)
    dropout_7 = Dropout(0.2)(conv_14)        
    conv_15 = Conv2D(64, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(dropout_7)

    conv_16 = Conv2DTranspose(32, (2,2), kernel_initializer='he_normal', strides=(2, 2), padding='same')(conv_15)
    concat_3 = concatenate([conv_16, conv_4])
    conv_17 = Conv2D(32, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(concat_3)
    dropout_8 = Dropout(0.1)(conv_17)        
    conv_18 = Conv2D(32, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(dropout_8)

    conv_19 = Conv2DTranspose(16, (2,2), kernel_initializer='he_normal', strides=(2, 2), padding='same')(conv_18)
    concat_3 = concatenate([conv_19, conv_2])
    conv_20 = Conv2D(16, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(concat_3)
    dropout_9 = Dropout(0.1)(conv_20)        
    conv_21 = Conv2D(16, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(dropout_9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv_21)

    model = Model(inputs=input_conv, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras_iou])
    model.summary()

    return model