import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Concatenate, Conv2D, Conv2DTranspose, Activation, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError
from tensorflow.keras.regularizers import l2
from kerastuner import HyperModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError
from tensorflow.keras.callbacks import ReduceLROnPlateau
from kerastuner.tuners import RandomSearch
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
import datetime

# Define input shape and number of bracelet types
input_shape = (256, 256, 3)  # Example input shape
num_bracelet_types = 50      # Number of different bracelet types


class Pix2PixHyperModel(HyperModel):
    def __init__(self, input_shape, mask_shape, num_bracelet_types):
        self.input_shape = input_shape
        self.mask_shape = mask_shape
        self.num_bracelet_types = num_bracelet_types

    def build_generator(self, hp):
        # Input image (model without jewelry)
        input_img = Input(shape=self.input_shape)

        # Segmentation mask for jewelry placement
        mask_input = Input(shape=self.mask_shape)

        # One-hot encoded bracelet type
        label_input = Input(shape=(self.num_bracelet_types,))
        label_embedding = Dense(np.prod(self.input_shape))(label_input)
        label_embedding = Reshape(self.input_shape)(label_embedding)

        # Combine image, mask, and label embedding
        combined_input = Concatenate()([input_img, mask_input, label_embedding])

        # Generator layers with hyperparameters
        # Encoder: Downsampling
        num_filters_d1 = hp.Choice('num_filters_d1', values=[64, 128, 256], default=64)
        d1 = Conv2D(num_filters_d1, (4, 4), strides=(2, 2), padding='same', kernel_regularizer=l2(0.01))(combined_input)
        d1 = LeakyReLU(alpha=0.2)(d1)

        num_filters_d2 = hp.Choice('num_filters_d2', values=[128, 256, 512], default=128)
        d2 = Conv2D(num_filters_d2, (4, 4), strides=(2, 2), padding='same', kernel_regularizer=l2(0.01))(d1)
        d2 = BatchNormalization()(d2)
        d2 = LeakyReLU(alpha=0.2)(d2)

        num_filters_d3 = hp.Choice('num_filters_d3', values=[256, 512, 1024], default=256)
        d3 = Conv2D(num_filters_d3, (4, 4), strides=(2, 2), padding='same', kernel_regularizer=l2(0.01))(d2)
        d3 = BatchNormalization()(d3)
        d3 = LeakyReLU(alpha=0.2)(d3)

        # Decoder: Upsampling with skip connections
        num_filters_u1 = hp.Choice('num_filters_u1', values=[128, 256, 512], default=128)
        u1 = Conv2DTranspose(num_filters_u1, (4, 4), strides=(2, 2), padding='same', kernel_regularizer=l2(0.01))(d3)
        u1 = BatchNormalization()(u1)
        u1 = Activation('relu')(u1)
        u1 = Concatenate()([u1, d2])  # Skip connection

        num_filters_u2 = hp.Choice('num_filters_u2', values=[64, 128, 256], default=64)
        u2 = Conv2DTranspose(num_filters_u2, (4, 4), strides=(2, 2), padding='same', kernel_regularizer=l2(0.01))(u1)
        u2 = BatchNormalization()(u2)
        u2 = Activation('relu')(u2)
        u2 = Concatenate()([u2, d1])  # Skip connection

        # Final layer
        output_img = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', activation='tanh', kernel_regularizer=l2(0.01))(u2)

        return Model([input_img, mask_input, label_input], output_img)


    def build_discriminator(self, hp):
        # Input image (model without jewelry)
        input_img = Input(shape=self.input_shape)

        # Segmentation mask for jewelry placement
        mask_input = Input(shape=self.mask_shape)

        # Target/generated image (model with bracelet)
        target_img = Input(shape=self.input_shape)

        # One-hot encoded bracelet type
        label_input = Input(shape=(self.num_bracelet_types,))
        label_embedding = Dense(np.prod(self.input_shape))(label_input)
        label_embedding = Reshape(self.input_shape)(label_embedding)

        # Concatenate inputs: input image, target image, mask, and label embedding
        combined_input = Concatenate()([input_img, target_img, mask_input, label_embedding])

        # Discriminator layers with hyperparameters
        num_filters_d1 = hp.Choice('disc_num_filters_d1', values=[64, 128, 256], default=64)
        d1 = Conv2D(num_filters_d1, (4, 4), strides=(2, 2), padding='same')(combined_input)
        d1 = LeakyReLU(alpha=0.2)(d1)

        num_filters_d2 = hp.Choice('disc_num_filters_d2', values=[128, 256, 512], default=128)
        d2 = Conv2D(num_filters_d2, (4, 4), strides=(2, 2), padding='same')(d1)
        d2 = BatchNormalization()(d2)
        d2 = LeakyReLU(alpha=0.2)(d2)

        num_filters_d3 = hp.Choice('disc_num_filters_d3', values=[256, 512, 1024], default=256)
        d3 = Conv2D(num_filters_d3, (4, 4), strides=(2, 2), padding='same')(d2)
        d3 = BatchNormalization()(d3)
        d3 = LeakyReLU(alpha=0.2)(d3)

        num_filters_d4 = hp.Choice('disc_num_filters_d4', values=[512, 1024, 2048], default=512)
        d4 = Conv2D(num_filters_d4, (4, 4), strides=(2, 2), padding='same')(d3)
        d4 = BatchNormalization()(d4)
        d4 = LeakyReLU(alpha=0.2)(d4)
        
        # Optional: Add a dropout layer
        dropout_rate = hp.Float('disc_dropout_rate', min_value=0.0, max_value=0.5, default=0.25, step=0.05)
        d4 = Dropout(dropout_rate)(d4)

        # Final layer
        output = Conv2D(1, (4, 4), padding='same')(d4)

        return Model([input_img, target_img, mask_input, label_input], output)

     def build_combined(self, generator, discriminator):
        # Build the combined model
        discriminator.trainable = False
        input_img = Input(shape=self.input_shape)
        label_input = Input(shape=(self.num_bracelet_types,))
        generated_img = generator([input_img, label_input])
        valid = discriminator([input_img, generated_img, label_input])
        combined_model = Model([input_img, label_input], [valid, generated_img])
        return combined_model

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


# Set up hyperparameter tuning (separate from the main training loop)
hypermodel = Pix2PixHyperModel(input_shape, mask_shape, num_bracelet_types)

# Learning rate scheduler callback
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

tuner = RandomSearch(
    hypermodel,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='hyperparam_tuning',
    project_name='pix2pix_tuning'
)

# Data Preparation (placeholder, implement based on your dataset)
#train_dataset, val_dataset = prepare_datasets()  # You need to define prepare_datasets function

# Perform hyperparameter search (use your train_dataset and val_dataset)
tuner.search(train_dataset, validation_data=val_dataset, epochs=20, callbacks=[early_stopping_callback, reduce_lr])

# Callbacks for the final model training
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
checkpoint_path = "model_checkpoints/cp.ckpt"
checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

# Final model training with the best hyperparameters
best_model.fit(train_dataset, epochs=50, validation_data=val_dataset, callbacks=[tensorboard_callback, early_stopping_callback, checkpoint_callback])


# Training loop placeholder
# (Include code to train the model using your dataset, labels, and training strategy)

# Training Loop (placeholder, implement based on your training strategy)
#train_model(best_model, train_dataset, val_dataset)  # You need to define train_model function
