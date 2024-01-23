import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Concatenate, Conv2D, Conv2DTranspose, Activation, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError
from kerastuner import HyperModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError


# Define input shape and number of bracelet types
input_shape = (256, 256, 3)  # Example input shape
num_bracelet_types = 50      # Number of different bracelet types


class Pix2PixHyperModel(HyperModel):
    def __init__(self, input_shape, mask_shape, num_bracelet_types):
        self.input_shape = input_shape
        self.mask_shape = mask_shape
        self.num_bracelet_types = num_bracelet_types

    def build_generator(input_shape, mask_shape, num_bracelet_types):
        # Input image (model without jewelry)
        input_img = Input(shape=input_shape)

        # Segmentation mask for jewelry placement
        mask_input = Input(shape=mask_shape)

        # One-hot encoded bracelet type
        label_input = Input(shape=(num_bracelet_types,))
        label_embedding = Dense(np.prod(input_shape))(label_input)
        label_embedding = Reshape(input_shape)(label_embedding)

        # Combine image, mask, and label embedding
        combined_input = Concatenate()([input_img, mask_input, label_embedding])

        # Generator layers
        # Encoder: Downsampling
        d1 = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_regularizer=l2(0.01))(combined_input)
        d1 = LeakyReLU(alpha=0.2)(d1)

        d2 = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_regularizer=l2(0.01))(d1)
        d2 = BatchNormalization()(d2)
        d2 = LeakyReLU(alpha=0.2)(d2)

        d3 = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_regularizer=l2(0.01))(d2)
        d3 = BatchNormalization()(d3)
        d3 = LeakyReLU(alpha=0.2)(d3)

        # Decoder: Upsampling with skip connections
        u1 = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_regularizer=l2(0.01))(d3)
        u1 = BatchNormalization()(u1)
        u1 = Activation('relu')(u1)
        u1 = Concatenate()([u1, d2])  # Skip connection

        u2 = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_regularizer=l2(0.01))(u1)
        u2 = BatchNormalization()(u2)
        u2 = Activation('relu')(u2)
        u2 = Concatenate()([u2, d1])  # Skip connection

        # Final layer
        output_img = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', activation='tanh', kernel_regularizer=l2(0.01))(u2)

        return Model([input_img, label_input], output_img)


    def build_discriminator(input_shape, mask_shape, num_bracelet_types):
        # Input image (model without bracelet)
        input_img = Input(shape=input_shape)

        # Segmentation mask for jewelry placement
        mask_input = Input(shape=mask_shape)

        # Target/generated image (model with bracelet)
        target_img = Input(shape=input_shape)

        # One-hot encoded bracelet type
        label_input = Input(shape=(num_bracelet_types,))
        label_embedding = Dense(np.prod(input_shape))(label_input)
        label_embedding = Reshape(input_shape)(label_embedding)

        # Concatenate inputs: input image, target image, mask, and label embedding
        combined_input = Concatenate()([input_img, target_img, mask_input, label_embedding])

        # Discriminator layers
        d1 = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(combined_input)
        d1 = LeakyReLU(alpha=0.2)(d1)

        d2 = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(d1)
        d2 = BatchNormalization()(d2)
        d2 = LeakyReLU(alpha=0.2)(d2)

        d3 = Conv2D(256, (4, 4), strides=(2, 2), padding='same')(d2)
        d3 = BatchNormalization()(d3)
        d3 = LeakyReLU(alpha=0.2)(d3)

        d4 = Conv2D(512, (4, 4), strides=(2, 2), padding='same')(d3)
        d4 = BatchNormalization()(d4)
        d4 = LeakyReLU(alpha=0.2)(d4)

        # Optional: Add a dropout layer
        d4 = Dropout(0.5)(d4)

        # Final layer
        output = Conv2D(1, (4, 4), padding='same')(d4)

        return Model([input_img, target_img, mask_input, label_input], output)

    def build(self, hp):
            generator = self.build_generator(hp)
            discriminator = self.build_discriminator(hp)
            combined_model = build_combined(generator, discriminator, self.input_shape, self.num_bracelet_types)
            combined_model.compile(
                optimizer=Adam(
                    lr=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
                ),
                loss=[BinaryCrossentropy(from_logits=True), MeanAbsoluteError()]
            )
            return combined_model

    # Instantiate and compile the generator and discriminator
    generator = build_generator(input_shape, num_bracelet_types)
    discriminator = build_discriminator(input_shape, num_bracelet_types)
    discriminator.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])

    # Define combined model for training the generator
    def build_combined(generator, discriminator, input_shape, num_bracelet_types):
        discriminator.trainable = False

        # Inputs
        input_img = Input(shape=input_shape)
        label_input = Input(shape=(num_bracelet_types,))

        # Generated image
        generated_img = generator([input_img, label_input])

        # Discriminator's decision
        valid = discriminator([input_img, generated_img, label_input])

        combined_model = Model([input_img, label_input], [valid, generated_img])
        combined_model.compile(loss=[BinaryCrossentropy(from_logits=True), MeanAbsoluteError()], optimizer=Adam(lr=0.0002, beta_1=0.5))

        return combined_model

    combined_model = build_combined(generator, discriminator, input_shape, num_bracelet_types)

    # Training loop placeholder
    # (Include code to train the model using your dataset, labels, and training strategy)
