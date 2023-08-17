import tensorflow as tf
from glob import glob
import os
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np


class DataSet:
    def __init__(self, dataset_dir : str, image_size : int=224, rand_aug_N : int=5, rand_aug_M : int=2):
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.class_names = os.listdir(self.dataset_dir)
        self.classes = [ x for x in range(len(self.class_names)) ]
        self.rand_aug = iaa.RandAugment(n=rand_aug_N, m=rand_aug_M)
        
        self.train_paths = []
        self.val_paths = []
        
        self.train_labels = []
        self.val_labels = []
        
        
    def __call__(self, validation_split : float=0.2, categorical : bool=True):
        
        for i in range(len(self.class_names)):
            files = [ f'{self.dataset_dir}/{self.class_names[i]}/' + x for x in os.listdir(f'{self.dataset_dir}/{self.class_names[i]}') ]
            labels = [ i for x in range(len(files)) ]
            
            # Split each class according to 'validation_split'
            lim = int(validation_split * len(files))
            
            self.train_paths.extend(files[lim:])
            self.val_paths.extend(files[:lim])
            self.train_labels.extend(labels[lim:])
            self.val_labels.extend(labels[:lim])
        
        # One Hot Encode the labels
        if categorical:
            self.train_labels = tf.keras.utils.to_categorical(self.train_labels)
            self.val_labels = tf.keras.utils.to_categorical(self.val_labels) if len(self.val_labels) != 0. else self.val_labels
            
        train_data = self.tf_dataset(self.train_paths, self.train_labels)
        
        # Return only train dataset if validation split is 0
        if validation_split == 0:
            return train_data.shuffle(buffer_size=1000, seed=225)
        
        val_data = self.tf_dataset(self.val_paths, self.val_labels)
        return train_data.shuffle(buffer_size=1000, seed=225), val_data.shuffle(buffer_size=1000, seed=225)


    # Read an image, resize it according to the given image shape.
    # Return image tensor scaled to [0, 1]
    def get_data(self, image, label):
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (self.image_size, self.image_size),)
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        return image, label


    # Used to create Tensorflow dataset from image paths and labels
    def tf_dataset(self, images, labels):
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.map(self.get_data)
        return dataset


    # SOURCE: https://keras.io/examples/vision/cutmix/
    def sample_beta_distribution(self, size, concentration_0=0.2, concentration_1=0.2):
        gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
        gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


    # SOURCE: https://keras.io/examples/vision/cutmix/
    @tf.function
    def get_box(self, lambda_value):
        cut_rat = tf.math.sqrt(1.0 - lambda_value)

        cut_w = self.image_size * cut_rat  # rw
        cut_w = tf.cast(cut_w, tf.int32)

        cut_h = self.image_size * cut_rat  # rh
        cut_h = tf.cast(cut_h, tf.int32)

        cut_x = tf.random.uniform((1,), minval=0, maxval=self.image_size, dtype=tf.int32)  # rx
        cut_y = tf.random.uniform((1,), minval=0, maxval=self.image_size, dtype=tf.int32)  # ry

        boundaryx1 = tf.clip_by_value(cut_x[0] - cut_w // 2, 0, self.image_size)
        boundaryy1 = tf.clip_by_value(cut_y[0] - cut_h // 2, 0, self.image_size)
        bbx2 = tf.clip_by_value(cut_x[0] + cut_w // 2, 0, self.image_size)
        bby2 = tf.clip_by_value(cut_y[0] + cut_h // 2, 0, self.image_size)

        target_h = bby2 - boundaryy1
        if target_h == 0:
            target_h += 1

        target_w = bbx2 - boundaryx1
        if target_w == 0:
            target_w += 1

        return boundaryx1, boundaryy1, target_h, target_w


    # SOURCE: https://keras.io/examples/vision/cutmix/
    @tf.function
    def cutmix(self, train_ds_one, train_ds_two):
        (image1, label1), (image2, label2) = train_ds_one, train_ds_two

        alpha = [0.25]
        beta = [0.25]

        # Get a sample from the Beta distribution
        lambda_value = self.sample_beta_distribution(1, alpha, beta)

        # Define Lambda
        lambda_value = lambda_value[0][0]

        # Get the bounding box offsets, heights and widths
        boundaryx1, boundaryy1, target_h, target_w = self.get_box(lambda_value)

        # Get a patch from the second image (`image2`)
        crop2 = tf.image.crop_to_bounding_box(
            image2, boundaryy1, boundaryx1, target_h, target_w
        )
        # Pad the `image2` patch (`crop2`) with the same offset
        image2 = tf.image.pad_to_bounding_box(
            crop2, boundaryy1, boundaryx1, self.image_size, self.image_size
        )
        # Get a patch from the first image (`image1`)
        crop1 = tf.image.crop_to_bounding_box(
            image1, boundaryy1, boundaryx1, target_h, target_w
        )
        # Pad the `image1` patch (`crop1`) with the same offset
        img1 = tf.image.pad_to_bounding_box(
            crop1, boundaryy1, boundaryx1, self.image_size, self.image_size
        )

        # Modify the first image by subtracting the patch from `image1`
        # (before applying the `image2` patch)
        image1 = image1 - img1
        # Add the modified `image1` and `image2`  together to get the CutMix image
        image = image1 + image2

        # Adjust Lambda in accordance to the pixel ration
        lambda_value = 1 - (target_w * target_h) / (self.image_size * self.image_size)
        lambda_value = tf.cast(lambda_value, tf.float32)

        # Combine the labels of both images
        label = lambda_value * label1 + (1 - lambda_value) * label2
        return image, label
    
    
    # SOURCE: https://keras.io/examples/vision/mixup/
    @tf.function
    def mix_up(self, ds_one, ds_two, alpha=0.2):
        # Unpack two datasets
        images_one, labels_one = ds_one
        images_two, labels_two = ds_two
        batch_size = tf.shape(images_one)[0]

        # Sample lambda and reshape it to do the mixup
        l = self.sample_beta_distribution(batch_size, alpha, alpha)
        x_l = tf.reshape(l, (batch_size, 1, 1, 1))
        y_l = tf.reshape(l, (batch_size, 1))

        # Perform mixup on both images and labels by combining a pair of images/labels
        # (one from each dataset) into one image/label
        images = images_one * x_l + images_two * (1 - x_l)
        labels = labels_one * y_l + labels_two * (1 - y_l)
        return (images, labels)
    

    # SOURCE: https://keras.io/examples/vision/randaugment/
    @tf.function
    def rand_augment(self, images):
        # Input to `augment()` is a TensorFlow tensor which
        # is not supported by `imgaug`. This is why we first
        # convert it to its `numpy` variant.
        images = images * 255.0
        images = tf.cast(images, tf.uint8)
        return self.rand_aug(images=images.numpy()) / 255.0


if __name__ == '__main__':
    # How to use the Dataset Class
    dataset = DataSet(dataset_dir='flower_images')
    
    # calling DataSet object returns train dataset and validation 
    # datast from a given image directory. Each class is 
    # split according to the given ratio.
    train_data, val_data = dataset(validation_split=0.2)
    
    
    ### Using CutMix
    # --------------------------------------------------------------------
    # Two datasets are required for CutMix.
    # SOURCE: https://keras.io/examples/vision/cutmix/
    t1 = train_data.map(lambda x, y: (x, y))
    t2 = train_data.map(lambda x, y: (x, y))
    
    td = tf.data.Dataset.zip((t1, t2))
    train_ds_cmu = td.map(dataset.cutmix, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)
    
    
    # Plot results
    image_batch, label_batch = next(iter(train_ds_cmu))
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.title(dataset.class_names[np.argmax(label_batch[i].numpy())])
        plt.imshow(image_batch[i])
        plt.axis("off")
    plt.suptitle('CutMix Augmentation')
    # plt.show()
    plt.savefig("figures/CutMix.png")
    
    
    ### Using MixUp
    # ---------------------------------------------------------------------
    # Two datasets are required for MixUp. NOTE: for MixUp we need to batch
    # first.
    # SOURCE: https://keras.io/examples/vision/mixup/
    t1 = train_data.map(lambda x, y: (x, y)).batch(32)
    t2 = train_data.map(lambda x, y: (x, y)).batch(32)
    td = tf.data.Dataset.zip((t1, t2))
    train_ds_mu = td.map(lambda ds_one, ds_two: dataset.mix_up(ds_one, ds_two, alpha=0.2), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    
    # Plot results
    image_batch, label_batch = next(iter(train_ds_mu))
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.title(dataset.class_names[np.argmax(label_batch[i].numpy())])
        plt.imshow(image_batch[i])
        plt.axis("off")
    plt.suptitle('MixUp Augmentation')
    plt.savefig("figures/MixUp.png")
    # plt.show()
    
    
    ### Using RandAugment
    # ---------------------------------------------------------------------
    # NOTE: we need to batch first for RandAugment. Only one dataset is required.
    train_ds_rand = train_data.batch(32).map(
        lambda x, y: (tf.py_function(dataset.rand_augment, [x], [tf.float32])[0], y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)
    
    # Plot results
    sample_images, _ = next(iter(train_ds_rand))
    plt.figure(figsize=(10, 10))
    for i, image in enumerate(sample_images[:9]):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image.numpy())
        plt.axis("off")
    plt.suptitle('Rand Augmentation')
    plt.savefig("figures/RandAugment.png")
    # plt.show()
    
    
    ### Using CutMix, MixUp and RandAugment together
    # ---------------------------------------------------------------------
    
    # First use CutMix, then Mixup, lastly RandAugment
    t1 = train_data.map(lambda x, y: (x, y))
    t2 = train_data.map(lambda x, y: (x, y))
    
    td = tf.data.Dataset.zip((t1, t2))
    train_ds_cmu1 = td.map(dataset.cutmix, num_parallel_calls=tf.data.AUTOTUNE).batch(32)
    train_ds_cmu2 = td.map(dataset.cutmix, num_parallel_calls=tf.data.AUTOTUNE).batch(32)
    td = tf.data.Dataset.zip((train_ds_cmu1, train_ds_cmu2))
    train_ds_cmu_mu = td.map(lambda train_ds_cmu1, train_ds_cmu2: dataset.mix_up(train_ds_cmu1, train_ds_cmu2, alpha=0.2), num_parallel_calls=tf.data.AUTOTUNE)
    
    train_ds_cmu_mu_rand = train_ds_cmu_mu.map(
        lambda x, y: (tf.py_function(dataset.rand_augment, [x], [tf.float32])[0], y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)
    
    # Plot results
    image_batch, label_batch = next(iter(train_ds_cmu_mu_rand))
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.title(dataset.class_names[np.argmax(label_batch[i].numpy())])
        plt.imshow(image_batch[i])
        plt.axis("off")
    plt.suptitle('CutMix + MixUp + Rand Augmentation')
    plt.savefig("figures/CutMix+MixUp+RandAugment.png")
    