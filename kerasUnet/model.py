from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout

from metrics import iou

def model():
    input_conv = Input((128, 128, 3))

    s = Lambda(lambda x: x / 255) (input_conv)

    # smaller
    conv_1 = Conv2D(16, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(s)
    conv_2 = Conv2D(16, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(conv_1)
    pooling_1 = MaxPooling2D((2,2))(conv_2)

    conv_3 = Conv2D(32, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(pooling_1)
    conv_4 = Conv2D(32, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(conv_3)
    pooling_2 = MaxPooling2D((2,2))(conv_4)

    conv_5 = Conv2D(64, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(pooling_2)
    conv_6 = Conv2D(64, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(conv_5)
    pooling_3 = MaxPooling2D((2,2))(conv_6)

    conv_7 = Conv2D(128, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(pooling_3)
    conv_8 = Conv2D(128, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(conv_7)
    pooling_4 = MaxPooling2D((2,2))(conv_8)

    # smallest
    conv_9 = Conv2D(256, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(pooling_4)
    conv_10 = Conv2D(256, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(conv_9)

    # larger
    conv_10 = Conv2DTranspose(128, (2,2), kernel_initializer='he_normal', strides=(2, 2), padding='same')(conv_10)
    concat_1 = concatenate([conv_10, conv_8])
    conv_11 = Conv2D(128, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(concat_1)
    conv_12 = Conv2D(128, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(conv_11)

    conv_13 = Conv2DTranspose(64, (2,2), kernel_initializer='he_normal', strides=(2, 2), padding='same')(conv_12)
    concat_2 = concatenate([conv_13, conv_6])
    conv_14 = Conv2D(64, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(concat_2)
    conv_15 = Conv2D(64, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(conv_14)

    conv_16 = Conv2DTranspose(32, (2,2), kernel_initializer='he_normal', strides=(2, 2), padding='same')(conv_15)
    concat_3 = concatenate([conv_16, conv_4])
    conv_17 = Conv2D(32, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(concat_3)
    conv_18 = Conv2D(32, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(conv_17)

    conv_19 = Conv2DTranspose(16, (2,2), kernel_initializer='he_normal', strides=(2, 2), padding='same')(conv_18)
    concat_3 = concatenate([conv_19, conv_2])
    conv_20 = Conv2D(16, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(concat_3)
    conv_21 = Conv2D(16, (3,3), kernel_initializer='he_normal', activation='relu', padding='same')(conv_20)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv_21)

    model = Model(inputs=input_conv, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_iou_metric])
    model.summary()

    return model