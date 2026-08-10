import random
import matplotlib.pyplot as plt
import tensorflow as tf


class UNet:
    def __init__(self, input_shape, num_classes, mini, maxi):
        
        """
        Initializes the UNet model with input parameters.

        Parameters:
        - input_shape (tuple): Shape of the input data (height, width, channels).
        - num_classes (int): Number of classes for segmentation.
        - mini (int): Minimum number of filters in the encoder layers.
        - maxi (int): Maximum number of filters in the encoder layers.
        """
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.mini = mini
        self.maxi = maxi
        self.model = self.build_model()    # Build the UNet model upon initialization

        
    def bottleneck(self, last_input, n_filters, padding="same"):
        # Define the bottleneck layer of the UNet model
        # Conv2D size and activation can be different. Look at the tensorflow website.
        conv_net = tf.keras.layers.Conv2D(n_filters, (3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(last_input)
        conv_net = tf.keras.layers.Conv2D(n_filters, (3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(conv_net)
        return conv_net

    def encoder_path(self, inputs, n_filters):
        # Define an encoder path of the UNet model
        # Conv2D size, activation, size of dropout and position of that can be different. Look at the tensorflow website.
        conv = tf.keras.layers.Conv2D(n_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        conv = tf.keras.layers.Dropout(0.2)(conv)
        conv = tf.keras.layers.Conv2D(n_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv)
        bf = tf.keras.layers.BatchNormalization()(conv)
        af = tf.keras.layers.ReLU()(bf)
        #MaxPooling2D size can be different also.
        next_layer = tf.keras.layers.MaxPooling2D((2, 2))(af)
        skip_connection = conv
        return next_layer, skip_connection

    def decoder_path(self, prev_layer, prev_skip_connection, n_filters):
        # Define an decoder path of the UNet model
        # Conv2D size, Conv2DTranspose size, activation, size of dropout and position of that and strides size can be different. Look at the tensorflow website.
        up_conv = tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(prev_layer)
        concat = tf.keras.layers.concatenate([up_conv, prev_skip_connection])
        conv = tf.keras.layers.Conv2D(n_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(concat)
        conv = tf.keras.layers.Dropout(0.2)(conv)
        conv = tf.keras.layers.Conv2D(n_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv)
        bf = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.ReLU()(bf)
        return conv

    def build_model(self):
        # Define the input layer based on the input shape
        inputs = tf.keras.layers.Input(shape=self.input_shape)

        # Build the UNet model architecture based on mini and maxi values
        if self.mini == 16 and self.maxi ==256:
            
            # Encoder path with decreasing resolution and increasing filters
            c1, s1 = self.encoder_path(inputs, 16)
            c2, s2 = self.encoder_path(c1, 32)
            c3, s3 = self.encoder_path(c2, 64)
            c4, s4 = self.encoder_path(c3, 128)
            
            # Bottleneck layer
            b1 = self.bottleneck(c4, 256)
            
            # Decoder path with increasing resolution and decreasing filters
            u1 = self.decoder_path(b1, s4, 128)
            u2 = self.decoder_path(u1, s3, 64)
            u3 = self.decoder_path(u2, s2, 32)
            u4 = self.decoder_path(u3, s1, 16)
            
            # Final convolutional layer to output segmentation mask
            outputs = tf.keras.layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid')(u4)
            
            # Define the model with inputs and outputs
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
        if self.mini == 16 and self.maxi ==512:
            
            c11, s11 = self.encoder_path(inputs, 16)
            c21, s21 = self.encoder_path(c11, 32)
            c31, s31 = self.encoder_path(c21, 64)
            c41, s41 = self.encoder_path(c31, 128)
            c51, s51 = self.encoder_path(c41, 256)
            b11 = self.bottleneck(c51, 512)
            u11 = self.decoder_path(b11, s51, 256)
            u21 = self.decoder_path(u11, s41, 128)
            u31 = self.decoder_path(u21, s31, 64)
            u41 = self.decoder_path(u31, s21, 32)
            u51 = self.decoder_path(u41, s11, 16)
            outputs = tf.keras.layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid')(u51)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            

        if self.mini == 32 and self.maxi ==256:

            c12, s12 = self.encoder_path(inputs, 32)
            c22, s22 = self.encoder_path(c12, 64)
            c32, s32 = self.encoder_path(c22, 128)
            b12 = self.bottleneck(c32, 256)
            u12 = self.decoder_path(b12, s32, 128)
            u22 = self.decoder_path(u12, s22, 64)
            u32 = self.decoder_path(u22, s12, 32)
            outputs = tf.keras.layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid')(u32)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
        if self.mini == 32 and self.maxi ==512:
            
            c13, s13 = self.encoder_path(inputs, 32)
            c23, s23 = self.encoder_path(c13, 64)
            c33, s33 = self.encoder_path(c23, 128)
            c43, s43 = self.encoder_path(c33, 256)
            c53, s53 = self.encoder_path(c43, 512)
            b13 = self.bottleneck(c53, 1024)
            u13 = self.decoder_path(b13, s53, 512)
            u23 = self.decoder_path(u13, s43, 256)
            u33 = self.decoder_path(u23, s33, 128)
            u43 = self.decoder_path(u33, s23, 64)
            u53 = self.decoder_path(u43, s13, 32)
            outputs = tf.keras.layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid')(u53)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
        if self.mini == 32 and self.maxi ==1024:
            
            c14, s14 = self.encoder_path(inputs, 32)
            c24, s24 = self.encoder_path(c14, 64)
            c34, s34 = self.encoder_path(c24, 128)
            c44, s44 = self.encoder_path(c34, 256)
            c54, s54 = self.encoder_path(c44, 512)
            b14 = self.bottleneck(c54, 1024)
            u14 = self.decoder_path(b14, s54, 512)
            u24 = self.decoder_path(u14, s44, 256)
            u34 = self.decoder_path(u24, s34, 128)
            u44 = self.decoder_path(u34, s24, 64)
            u54 = self.decoder_path(u44, s14, 32)
            outputs = tf.keras.layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid')(u54)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            

        if self.mini == 64 and self.maxi ==256:

            c15, s15 = self.encoder_path(inputs, 64)
            c25, s25 = self.encoder_path(c15, 128)
            b15 = self.bottleneck(c25, 256)
            u15 = self.decoder_path(b15, s25, 128)
            u25 = self.decoder_path(u15, s15, 64)
            outputs = tf.keras.layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid')(u25)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
        if self.mini == 64 and self.maxi ==512:

            c16, s16 = self.encoder_path(inputs, 64)
            c26, s26 = self.encoder_path(c16, 128)
            c36, s36 = self.encoder_path(c26, 256)
            b16 = self.bottleneck(c36, 512)
            u16 = self.decoder_path(b16, s36, 256)
            u26 = self.decoder_path(u16, s26, 128)
            u36 = self.decoder_path(u26, s16, 64)
            outputs = tf.keras.layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid')(u36)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
        if self.mini == 64 and self.maxi ==1024:

            c17, s17 = self.encoder_path(inputs, 64)
            c27, s27 = self.encoder_path(c17, 128)
            c37, s37 = self.encoder_path(c27, 256)
            c47, s47 = self.encoder_path(c37, 512)
            b17 = self.bottleneck(c47, 1024)
            u17 = self.decoder_path(b17, s47, 512)
            u27 = self.decoder_path(u17, s37, 256)
            u37 = self.decoder_path(u27, s27, 128)
            u47 = self.decoder_path(u37, s17, 64)
            outputs = tf.keras.layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid')(u47)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        return model

    def get_model(self):
        return self.model





