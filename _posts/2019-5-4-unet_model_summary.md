### Unet framework in Keras Implement
```python
from keras.layers import *
from kereas.models import *
from kereas.optimizers import *

def build_unet(input_shape, pretrained_weight=None):
    # inputs layer
    inputs = Input(shape=input_shape)
    # conv in 1th phrase
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    # conv in 2th phrase
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    # conv in 3th phrase
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    # conv in 4th phrase
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(drop4)
    
    # conv in 5th phrase and not pooling
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
 
    # upsampling in 6th phrase
    up6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling(size=(2,2))(drop5))
    merge6 = concatenate([up6, pool4], axis=-1)
    conv6 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
 
    # upsampling in 7th phrase
    up7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling(size=(2,2))(conv6))
    merge7 = concatenate([up7, pool3], axis=-1)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
 
    # upsampling in 8th phrase
    up8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling(size=(2,2))(conv7))
    merge8 = concatenate([up8, pool2], axis=-1)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    # upsampling in 9th phrase
    up9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling(size=(2,2))(conv8))
    merge9 = concatenate([up9, pool1], axis=-1)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
 
    # conv in output phrase
    outputs = Conv2D(1,1, activation='sigmoid')(conv9)

    unet_model = Model(inputs, outputs)
    unet_model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metric=['accuracy'])
    print(unet_model.summary())
    if pretrained_weight:
        unet_model.load_weights(pretrained_weight)

    return unet_model
```
