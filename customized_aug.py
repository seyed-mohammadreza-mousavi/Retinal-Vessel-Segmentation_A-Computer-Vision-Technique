import keras
from keras import layers
num_images = len(train_images_preprocessed)
num_augmentations_per_image = 5  # Specify the number of augmentations per image

def custom_data_generator(image, groundtruth):
    # Define augmentation layers
    augmentation_layers = keras.Sequential(
        [
            layers.RandomRotation(factor=0.1),  # Rotation within -45 to +45 degrees
            #layers.RandomTranslation(height_factor=0.01, width_factor=0.01),  # Random translation
            layers.RandomFlip(),  # Random horizontal flip
        ],
        name="img_augmentation",
    )
    image = tf.cast(image, tf.int64)
    groundtruth = tf.expand_dims(tf.cast(groundtruth, tf.int64), axis=3)
    input_tensor = tf.concat([image, groundtruth], axis=3)

    # Apply the same augmentation to both the image and groundtruth
    augmented_tensor = augmentation_layers(input_tensor)

    augmented_image = augmented_tensor[:, :, :, :3]
    augmented_groundtruth = augmented_tensor[:, :, :, 3:]

    return augmented_image, augmented_groundtruth

# Create a list to store augmented images
augmented_pairs = []
# Apply augmentations to each image in the dataset
for i, j in tqdm(img_lbl_pair, "Augmenting"):
    augmented_pairs.append((i, j))
    for _ in range(num_augmentations_per_image):
        image = tf.expand_dims(i, axis=0)
        g = tf.expand_dims(j, axis=0)
        augmented_image, augmented_groundtruth = custom_data_generator(image, g)
        #image = tf.expand_dims(i, axis=0)
        #g = tf.expand_dims(j, axis=0)
        #augmented_image=img_augmentation(image)
        #augmented_groundtruth=img_augmentation(g)
        augmented_pairs.append((augmented_image[0],augmented_groundtruth[0]))