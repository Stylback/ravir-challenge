#-------------------
# Prerequisite modules

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, Conv2DTranspose, concatenate, multiply
from tensorflow.keras import Model

#--------------------
# Convolution

def conv_block(base, batchnormalization, bool_first, img_w, img_h, img_c, input_layer):
    
    if bool_first:
        first = Conv2D(filters=base, input_shape=(img_w, img_h, img_c),
                       kernel_size=(3,3), padding='same')(input_layer)
    else:
        first = Conv2D(filters=base, kernel_size=(3,3), padding='same')(input_layer)
        
    if batchnormalization:
        second = BatchNormalization()(first)
        third = Activation('relu')(second)
        fourth = Conv2D(filters=base, kernel_size=(3,3), padding='same')(third)
        fifth = BatchNormalization()(fourth)
        last = Activation('relu')(fifth)
    else:
        second = Activation('relu')(first)
        third = Conv2D(filters=base, kernel_size=(3,3), padding='same')(second)
        last = Activation('relu')(third)
    return last

#--------------------
# Contraction

def encoder_block(bool_first, batchnormalization, dropout, dropout_rate, base, img_w, img_h, img_c, input_layer):
    
    block_output = conv_block(base, batchnormalization, bool_first, img_w, img_h, img_c, input_layer)
    contraction_out = MaxPooling2D(pool_size=(2,2))(block_output)
    
    if dropout:
        contraction_out = Dropout(dropout_rate)(contraction_out)

    return block_output, contraction_out

#--------------------
# Expansion

def decoder_block(batchnormalization, dropout, dropout_rate, base, img_w, img_h, img_c, input_layer, contraction_corr):
    
    conc = concatenate([Conv2DTranspose(filters=base, kernel_size=(2,2), strides=(2,2), padding='same')
                        (input_layer), contraction_corr], axis=3)
    if dropout:
        dropout = Dropout(dropout_rate)(conc)
        expansion_out = conv_block(base, batchnormalization, False, img_w, img_h, img_c, dropout)
        
    else:
        expansion_out = conv_block(base, batchnormalization, False, img_w, img_h, img_c, conc)
    
    return expansion_out

#--------------------
# UNet model with weight maps

def get_weighted_unet(weighted_input, base, batchnormalization, dropout, dropout_rate, img_w, img_h, img_c, n_classes=1):
    
    inputs_layer = weighted_input[0]
    weight_map = weighted_input[1]
    
    #Contraction
    corr1, cont1 = encoder_block(True, batchnormalization, dropout, dropout_rate, base, img_w, img_h, img_c, inputs_layer)
    corr2, cont2 = encoder_block(False, batchnormalization, dropout, dropout_rate, 2*base, img_w, img_h, img_c, cont1)
    corr3, cont3 = encoder_block(False, batchnormalization, dropout, dropout_rate, 4*base, img_w, img_h, img_c, cont2)
    corr4, cont4 = encoder_block(False, batchnormalization, dropout, dropout_rate, 8*base, img_w, img_h, img_c, cont3)
      
    #Bottleneck
    bottleneck_out = conv_block(16*base, batchnormalization, False, img_w, img_h, img_c, cont4)
    
    #Expansion
    exp1 = decoder_block(batchnormalization, dropout, dropout_rate, 8*base, img_w, img_h, img_c, bottleneck_out, corr4)
    exp2 = decoder_block(batchnormalization, dropout, dropout_rate, 4*base, img_w, img_h, img_c, exp1, corr3)
    exp3 = decoder_block(batchnormalization, dropout, dropout_rate, 2*base, img_w, img_h, img_c, exp2, corr2)
    exp4 = decoder_block(batchnormalization, dropout, dropout_rate, base, img_w, img_h, img_c, exp3, corr1)
    
    #Final block
    if n_classes > 1:
        out = Conv2D(filters=n_classes, kernel_size=(1,1), padding='same', activation='softmax')(exp4)
        out = multiply([out, weight_map])
    else:
        out = Conv2D(filters=1, kernel_size=(1,1), padding='same', activation='sigmoid')(exp4)
        out = multiply([out, weight_map])
        
    model = Model(inputs=weighted_input, outputs=out, name='Weighted UNet')
    model.summary()
    
    return model