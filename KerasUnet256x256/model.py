from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout

def model():
    input_conv = Input((256, 256, 3))

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

    conv_9 = Conv2D(256, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(pooling_4)
    dropout_5 = Dropout(0.3)(conv_9)    
    conv_10 = Conv2D(256, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(dropout_5)
    pooling_6 = MaxPooling2D((2,2))(conv_10)

    # smallest
    conv_11 = Conv2D(512, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(pooling_6)
    dropout_6 = Dropout(0.4)(conv_11)    
    conv_12 = Conv2D(512, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(dropout_6)

    # larger
    conv_13 = Conv2DTranspose(256, (2,2), kernel_initializer='he_normal', strides=(2, 2), padding='same')(conv_12)
    concat_1 = concatenate([conv_13, conv_10])
    conv_14 = Conv2D(256, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(concat_1)
    dropout_7 = Dropout(0.3)(conv_14)    
    conv_15 = Conv2D(256, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(dropout_7)

    conv_16 = Conv2DTranspose(128, (2,2), kernel_initializer='he_normal', strides=(2, 2), padding='same')(conv_15)
    concat_2 = concatenate([conv_16, conv_8])
    conv_17 = Conv2D(128, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(concat_2)
    dropout_8 = Dropout(0.2)(conv_17)    
    conv_18 = Conv2D(128, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(dropout_8)

    conv_19 = Conv2DTranspose(64, (2,2), kernel_initializer='he_normal', strides=(2, 2), padding='same')(conv_18)
    concat_3 = concatenate([conv_19, conv_6])
    conv_20 = Conv2D(64, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(concat_3)
    dropout_9 = Dropout(0.2)(conv_20)
    conv_21 = Conv2D(64, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(dropout_9)

    conv_22 = Conv2DTranspose(32, (2,2), kernel_initializer='he_normal', strides=(2, 2), padding='same')(conv_21)
    concat_4 = concatenate([conv_22, conv_4])
    conv_23 = Conv2D(32, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(concat_4)
    dropout_10 = Dropout(0.1)(conv_23)
    conv_24 = Conv2D(32, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(dropout_10)

    conv_25 = Conv2DTranspose(16, (2,2), kernel_initializer='he_normal', strides=(2, 2), padding='same')(conv_24)
    concat_5 = concatenate([conv_25, conv_2])
    conv_26 = Conv2D(16, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(concat_5)
    dropout_11 = Dropout(0.1)(conv_26)
    conv_27 = Conv2D(16, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(dropout_11)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv_27)

    model = Model(inputs=input_conv, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()

    return model