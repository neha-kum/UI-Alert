# Libraries
'''
import os
import cv2
import csv
import numpy as np
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# Constants
MODEL_PATH = "./UNetRoadSegmentation_SIH_T1.keras"

# Others
import warnings
warnings.filterwarnings("ignore")

# ************************************* Model Section *********************************

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
N_IMAGE_CHANNELS = 3

class EncoderBlock(layers.Layer):

    def __init__(self, filters: int, max_pool: bool=True, rate=0.2, **kwargs) -> None:
        super().__init__(**kwargs)

        # Params
        self.rate = rate
        self.filters = filters
        self.max_pool = max_pool

        # Layers : Initialize the model layers that will be later called
        self.max_pooling = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))
        self.conv1 = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal'
        )
        self.conv2 = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal'
        )
        self.drop = layers.Dropout(rate)
        self.bn = layers.BatchNormalization()

    def call(self, X, **kwargs):

        X = self.bn(X)
        X = self.conv1(X)
        X = self.drop(X)
        X = self.conv2(X)

        # Apply Max Pooling if required
        if self.max_pool:
            y = self.max_pooling(X)
            return y, X
        else:
            return X

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'max_pool': self.max_pool,
            'rate': self.rate
        })
        return config

    def __repr__(self):
        return f"{self.__class__.__name__}(F={self.filters}, Pooling={self.max_pool})"

class DecoderBlock(layers.Layer):

    def __init__(self, filters: int, rate: float = 0.2, **kwargs):
        super().__init__(**kwargs)

        self.filters = filters
        self.rate = rate

        # Initialize the model layers
        self.convT = layers.Conv2DTranspose(
            filters = filters,
            kernel_size = 3,
            strides = 2,
            padding = 'same',
            activation = 'relu',
            kernel_initializer = 'he_normal'
        )
        self.bn = layers.BatchNormalization()
        self.net = EncoderBlock(filters = filters, rate = rate, max_pool = False)

    def call(self, inputs, **kwargs):

        # Get both the inputs
        X, skip_X = inputs

        # Up-sample the skip connection
        X = self.bn(X)
        X = self.convT(X)

        # Concatenate both inputs
        X = layers.Concatenate(axis=-1)([X, skip_X])
        X = self.net(X)

        return X

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'rate': self.rate,
        })
        return config

    def __repr__(self):
        return f"{self.__class__.__name__}(F={self.filters}, rate={self.rate})"

def dice_coeff(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float=1.0) -> tf.Tensor:
    """Compute the Dice coefficient between predicted and true masks."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = tf.reduce_mean((2.0 * intersection + smooth) / (union + smooth), axis=0)

    return tf.cast(dice, tf.float32)

def load_model(model_path=MODEL_PATH):
    """Load a pre-trained model from the given path."""
    model = keras.models.load_model(model_path, custom_objects={
        "EncoderBlock": EncoderBlock,
        "DecoderBlock": DecoderBlock,
        "dice_coeff": dice_coeff,
    })

    # Freeze Model Weights
    model.trainable = False

    return model

# ************************************* Model Working Section  *********************************
def generate_masks(images, model, mask_threshold=0.25):
    """Generate binary masks for the images using the model."""
    images = tf.data.Dataset.from_tensor_slices((images))
    images = images.batch(batch_size=16).prefetch(buffer_size=tf.data.AUTOTUNE)
    binary_masks = model.predict(images, verbose=1)
    rot_images = images.map(lambda x: tf.image.rot90(x, k=1))
    gen_masks_rotated = model.predict(rot_images, verbose=1)
    binary_masks_rotated = tf.image.rot90(gen_masks_rotated, k=3)
    combined_masks_rot = binary_masks_rotated + binary_masks
    combined_masks_rot = tf.cast(combined_masks_rot > mask_threshold, dtype=tf.float32)
    return combined_masks_rot

def save_masks(masks, output_dir, input_dir):
    """Save the binary masks to the given directory."""
    filepaths = glob(input_dir + '/*')
    mask_paths = [path.split('\\')[-1] for path in filepaths]
    mask_paths = [path.replace('.', '_mask.') for path in mask_paths]
    mask_paths = [os.path.join(output_dir, path) for path in mask_paths]

    # Save the generated Masks
    for mask, path in tqdm(zip(masks, mask_paths), desc="Saving"):
        mask = tf.squeeze(mask)
        plt.imsave(fname=path, arr=mask, cmap="gray")

# ************************************* Drone Image Loading Section  *********************************
def load_drone_image(image_path):
    """Load a single drone image, crop and resize it."""
    image = tf.io.read_file(filename = image_path)
    processed_image = tf.image.decode_jpeg(contents = image, channels = N_IMAGE_CHANNELS)
    processed_image = tf.image.convert_image_dtype(image = processed_image, dtype = tf.float32)

    # Crop to square and resize
    image_shape = tf.shape(processed_image)
    height = image_shape[0]
    width = image_shape[1]
    min_dim = tf.minimum(height, width)

    if (height != IMAGE_HEIGHT) or (width != IMAGE_WIDTH):

        processed_image = tf.image.resize_with_crop_or_pad(processed_image, target_height=min_dim, target_width=min_dim)
        processed_image = tf.image.resize(images=processed_image, size=(IMAGE_WIDTH, IMAGE_HEIGHT))

    processed_image = tf.clip_by_value(processed_image, clip_value_min = 0.0, clip_value_max = 1.0)
    return tf.cast(processed_image, dtype = tf.float32)

def load_drone_images(filepaths):
    """Load multiple drone images from filepaths into a TensorFlow dataset."""
    images_numpy = np.empty(shape=(len(filepaths), IMAGE_WIDTH, IMAGE_HEIGHT, N_IMAGE_CHANNELS), dtype=np.float32)
    index = 0
    for image_path in tqdm(filepaths, desc="Loading"):
        image = load_drone_image(image_path)
        images_numpy[index] = image
        index+=1
    return images_numpy

def save_processed_images_mask(images, masks, image_output_dir, mask_output_dir, filepaths):
    image_paths = [os.path.join(image_output_dir, os.path.basename(path)) for path in filepaths]
    mask_paths = [os.path.join(mask_output_dir, os.path.basename(path)) for path in filepaths]

    # Save the generated Masks
    for image, mask, image_path, mask_path in tqdm(zip(images, masks, image_paths, mask_paths), desc="Saving"):
        plt.imsave(fname=image_path, arr=image)
        plt.imsave(fname=mask_path, arr=tf.squeeze(mask), cmap="gray")

# ************************************* ALERT Generation *********************************
def compute_impact_ratio(mask, total_pixels):
    white_pixels = np.sum(mask)
    return white_pixels / total_pixels

def compute_new_and_removed_roads(original_mask, new_mask, total_pixels=256*256):
    # Ensure both masks are binary
    _, original_binary = cv2.threshold(original_mask, 0.5, 1, cv2.THRESH_BINARY)
    _, new_binary = cv2.threshold(new_mask.numpy(), 0.5, 1, cv2.THRESH_BINARY)

    # Perform morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    original_binary = cv2.morphologyEx(original_binary, cv2.MORPH_CLOSE, kernel)
    new_binary = cv2.morphologyEx(new_binary, cv2.MORPH_CLOSE, kernel)

    # Compute new roads & removed roads
    new_roads = np.maximum(new_binary - original_binary, 0)
    removed_roads = np.maximum(original_binary - new_binary, 0)

    # Compute Impact Ratio
    ir_new = compute_impact_ratio(new_roads, total_pixels)
    ir_removed = compute_impact_ratio(removed_roads, total_pixels)
    tir = ir_new + ir_removed

    return tir, new_roads, removed_roads, ir_new, ir_removed
 
def show_detailed_changes(original_mask, new_mask, new_roads, removed_roads, original_image, new_image, tir, ir_new, ir_removed, alert_image_path=None):
    """
    Display detailed changes between masks and TIR.
    """
    plt.figure(figsize=(20, 8))

    plt.subplot(2, 3, 1)
    plt.title("Original Mask")
    plt.imshow(original_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("New Mask")
    plt.imshow(new_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("New Image")
    plt.imshow(new_image)
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("New Roads Constructed")
    plt.imshow(new_roads, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title("Previous Roads Removed")
    plt.imshow(removed_roads, cmap='gray')
    plt.axis('off')

    plt.suptitle(f"TIR (Total Impact Ratio): {tir:.2f}", fontsize=16)
    plt.tight_layout()

    if alert_image_path is not None:
        plt.savefig(alert_image_path)
        filename = os.path.basename(alert_image_path)
        alert_dir = os.path.dirname(alert_image_path)
        csv_filepath = os.path.join(alert_dir, 'alert.csv')

        # Step 1: Check if the CSV file exists
        if not os.path.exists(csv_filepath):
            # If CSV doesn't exist, create it with the header and the new data
            with open(csv_filepath, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['file_name', 'TIR', 'ir_new', 'ir_removed'])
                writer.writerow([filename, tir, ir_new, ir_removed])  # Add the new row
        else:
            # Step 2: If CSV exists, check for existing entries
            existing_data = []
            file_exists_in_csv = False

            with open(csv_filepath, 'r', newline='') as file:
                reader = csv.reader(file)
                existing_data = list(reader)

            # Check if the file_name exists, if so, update the row
            for row in existing_data:
                if row and row[0] == filename:  # If file_name exists
                    file_exists_in_csv = True
                    row[1] = tir  # Update TIR value
                    row[2] = ir_new  # Update ir_new value
                    row[3] = ir_removed  # Update ir_removed value
                    break

            # If file_name is not found, append new data
            if not file_exists_in_csv:
                existing_data.append([filename, tir, ir_new, ir_removed])

            # Write the updated data back to the CSV
            with open(csv_filepath, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(existing_data)

    plt.show()

# ************************************* Extra *********************************
def clear_console():
    # For Windows
    if os.name == 'nt':
        os.system('cls')
    # For macOS and Linux (posix)
    else:
        os.system('clear')

def load_prev_imgs_masks(file_names, image_dir, mask_dir):
    """Load multiple drone images from filepaths into a TensorFlow dataset."""
    mask_paths = [os.path.join(mask_dir, name) for name in file_names]
    image_paths = [os.path.join(image_dir, name) for name in file_names]

    og_masks = np.empty(shape=(len(mask_paths), IMAGE_WIDTH, IMAGE_HEIGHT, 1), dtype=np.float32)
    og_images = np.empty(shape=(len(mask_paths), IMAGE_WIDTH, IMAGE_HEIGHT, 3), dtype=np.float32)

    index = 0
    for mask_path, image_path in zip(mask_paths, image_paths):

        mask = load_drone_image(mask_path)[:,:,:1]
        image = load_drone_image(image_path)
        og_masks[index] = mask
        og_images[index] = image
        index+=1
        
    return mask_paths, og_masks, og_images



'''