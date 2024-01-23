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
import os
import random
import shutil
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import MeanAbsoluteError

###########################################################################################
################               Data preparation and splitting               ###############
###########################################################################################

# Function to apply augmentations to an image
def apply_augmentation(image):
    augmented_images = []

    # 1. Horizontal Flip
    augmented_images.append(ImageOps.mirror(image))

    # 2. Rotation
    augmented_images.append(image.rotate(random.uniform(-15, 15)))

    # 3. Scaling/Zooming
    scale = random.uniform(0.9, 1.1)
    width, height = image.size
    image_zoomed = image.resize((int(scale * width), int(scale * height)), Image.ANTIALIAS)
    image_zoomed = ImageOps.fit(image_zoomed, (width, height), centering=(0.5, 0.5))
    augmented_images.append(image_zoomed)

    # 4. Brightness Adjustment
    enhancer = ImageEnhance.Brightness(image)
    augmented_images.append(enhancer.enhance(random.uniform(0.8, 1.2)))

    # 5. Contrast Adjustment
    enhancer = ImageEnhance.Contrast(image)
    augmented_images.append(enhancer.enhance(random.uniform(0.8, 1.2)))

    # 6. Color Jitter
    enhancer = ImageEnhance.Color(image)
    augmented_images.append(enhancer.enhance(random.uniform(0.8, 1.2)))

    # 7. Cropping
    left = width * 0.1
    top = height * 0.1
    right = width * 0.9
    bottom = height * 0.9
    augmented_images.append(image.crop((left, top, right, bottom)).resize((width, height)))

    # 8. Gaussian Blur
    augmented_images.append(image.filter(ImageFilter.GaussianBlur(radius=2)))

    # 9. Noise Injection
    np_image = np.array(image)
    noise = np.random.normal(0, 25, np_image.shape)
    np_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
    augmented_images.append(Image.fromarray(np_image))

    return augmented_images

# Function to combine input and target images side by side
def combine_images(input_image, target_image):
    # Ensure both images are the same size
    target_image = target_image.resize(input_image.size)

    # Create a new image with double width and the same height as the input images
    combined_image = Image.new('RGB', (2 * input_image.width, input_image.height))

    # Paste the input and target images side by side
    combined_image.paste(input_image, (0, 0))
    combined_image.paste(target_image, (input_image.width, 0))

    return combined_image

# Directory paths
input_images_dir = 'path/to/input_images'
target_images_dir = 'path/to/target_images'
combined_images_dir = 'path/to/combined_images'

# Ensure output directory exists
os.makedirs(combined_images_dir, exist_ok=True)

# Iterate over all image pairs and apply augmentations
for filename in os.listdir(input_images_dir):
    input_image_path = os.path.join(input_images_dir, filename)
    target_image_path = os.path.join(target_images_dir, filename)

    input_image = Image.open(input_image_path)
    target_image = Image.open(target_image_path)

    augmented_input_images = apply_augmentation(input_image)

    for i, aug_input_image in enumerate(augmented_input_images):
        combined_image = combine_images(aug_input_image, target_image)
        combined_image_path = os.path.join(combined_images_dir, f"{filename.split('.')[0]}_aug{i}.jpg")
        combined_image.save(combined_image_path)

# Data Splitting
# Define the split ratios
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# Get all the combined image filenames
all_filenames = os.listdir(combined_images_dir)
np.random.shuffle(all_filenames)  # Randomly shuffle the list

# Calculate the number of images for each set
total_images = len(all_filenames)
train_count = int(total_images * train_ratio)
val_count = int(total_images * val_ratio)

# Split the filenames
train_filenames = all_filenames[:train_count]
val_filenames = all_filenames[train_count:train_count + val_count]
test_filenames = all_filenames[train_count + val_count:]

# Function to copy files to the respective directories
def copy_files(filenames, source_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for filename in filenames:
        shutil.copy(os.path.join(source_dir, filename), os.path.join(dest_dir, filename))

# Directory paths for split data
train_dir = 'path/to/train_data'
val_dir = 'path/to/val_data'
test_dir = 'path/to/test_data'

# Copy the files to the respective directories
copy_files(train_filenames, combined_images_dir, train_dir)
copy_files(val_filenames, combined_images_dir, val_dir)
copy_files(test_filenames, combined_images_dir, test_dir)

def prepare_datasets(train_dir, val_dir, image_size=(256, 256), batch_size=32):
    """
    Prepare training and validation datasets.
    
    Args:
    - train_dir: Directory containing training data.
    - val_dir: Directory containing validation data.
    - image_size: Tuple specifying the size of the images.
    - batch_size: Batch size for training.

    Returns:
    - train_dataset: A tf.data.Dataset object for training.
    - val_dataset: A tf.data.Dataset object for validation.
    """
    # Create ImageDataGenerators for data augmentation (for training) and only rescaling (for validation)
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Normalize images
        # Add any additional augmentation parameters if needed
    )
    val_datagen = ImageDataGenerator(rescale=1./255)  # Normalize images

    # Create iterators for the training and validation datasets
    train_dataset = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=None  # Set to None for unsupervised learning
    )
    val_dataset = val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=None  # Set to None for unsupervised learning
    )

    return train_dataset, val_dataset
# Prepare datasets
train_dataset, val_dataset = prepare_datasets(train_dir, val_dir)

###########################################################################################
################        HyperParameter Tuning and Training Loop             ###############
###########################################################################################
    
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
        combined_model = self.build_combined(generator, discriminator)

        # Compile the Generator model
        generator.compile(
            optimizer=Adam(lr=hp.Float('generator_learning_rate', 1e-4, 1e-2, sampling='log')),
            loss=MeanAbsoluteError(),
            metrics=[MeanAbsoluteError()]
        )

        # Compile the Discriminator model
        discriminator.compile(
            optimizer=Adam(lr=hp.Float('discriminator_learning_rate', 1e-4, 1e-2, sampling='log')),
            loss=BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Compile the Combined model
        combined_model.compile(
            optimizer=Adam(lr=hp.Float('combined_model_learning_rate', 1e-4, 1e-2, sampling='log')),
            loss=[BinaryCrossentropy(from_logits=True), MeanAbsoluteError()],
            metrics=[MeanAbsoluteError()]
        )

        return combined_model


###########################################################################################
################                training the pix2pix Model                  ###############
###########################################################################################
def train_pix2pix(generator, discriminator, combined_model, train_dataset, val_dataset, epochs=50):
    # Initialize metric for L1 loss
    l1_loss_metric = MeanAbsoluteError()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        for step, (input_images, target_images) in enumerate(train_dataset):
            # Here, input_images and target_images should be the corresponding parts of your data

            # Generate fake images
            generated_images = generator.predict(input_images)

            # Create labels for real and fake images
            valid = np.ones((input_images.shape[0],) + discriminator.output_shape[1:])
            fake = np.zeros((input_images.shape[0],) + discriminator.output_shape[1:])

            # Train discriminator
            d_loss_real = discriminator.train_on_batch([input_images, target_images], valid)
            d_loss_fake = discriminator.train_on_batch([input_images, generated_images], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            g_loss = combined_model.train_on_batch([input_images, target_images], [valid, target_images])

            print(f"Step {step + 1}: D Loss: {d_loss}, G Loss: {g_loss}")

        # Validation step at the end of each epoch
        l1_loss_metric.reset_states()
        for val_input_images, val_target_images in val_dataset:
            val_generated_images = generator.predict(val_input_images)
            l1_loss_metric.update_state(val_target_images, val_generated_images)
        
        val_l1_loss = l1_loss_metric.result().numpy()
        print(f"Validation L1 Loss: {val_l1_loss}")
        

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

# Perform hyperparameter search (use your train_dataset and val_dataset)
tuner.search(train_dataset, validation_data=val_dataset, epochs=20, callbacks=[early_stopping_callback, reduce_lr])

# Retrieve the best hyperparameters and build final models
best_hyperparameters = tuner.get_best_hyperparameters()[0]
generator = build_generator(best_hyperparameters)
discriminator = build_discriminator(best_hyperparameters)
combined_model = build_combined(generator, discriminator)


# Callbacks for the final model training
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
checkpoint_path = "model_checkpoints/cp.ckpt"
checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

# Final model training with the best hyperparameters
best_model.fit(train_dataset, epochs=50, validation_data=val_dataset, callbacks=[tensorboard_callback, early_stopping_callback, checkpoint_callback])

# Train the Pix2Pix model
train_pix2pix(generator, discriminator, combined_model, train_dataset, val_dataset, epochs=50)

# Save the model after training
generator.save('generator_model.h5')
discriminator.save('discriminator_model.h5')
combined_model.save('combined_model.h5')