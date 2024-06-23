import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
import os

imgSize = 128

def standardize_dataset(x, axis=None):
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.sqrt(((x - mean)**2).mean(axis=axis, keepdims=True))
    return (x - mean) / std

def add_gaussian_noise(X, mean=0, std=1):
    """Returns a copy of X with Gaussian noise."""
    return X.copy() + std * np.random.standard_normal(X.shape) + mean

class DataManager:
    def __init__(self, patch_size=32):
        self.X = None
        self.training_set_size = None
        self.patch_size = patch_size
        self.load_data()

    def load_data(self):
        patcheData = []
        
        for files in os.listdir('src/celeba'):
            if files.endswith('.jpg') or files.endswith('png'):
                img = (np.array(Image.open(os.path.join('src/celeba', files)).resize((imgSize, imgSize)))) 
                
                height, width, channels = img.shape
                for y in range(0, height - self.patch_size + 1, self.patch_size):
                    for x in range(0, width - self.patch_size + 1, self.patch_size):
                        patch = img[y:y+self.patch_size, x:x+self.patch_size]
                        patcheData.append(patch)
                
        data = np.array(patcheData).astype(np.float32)
        np.random.shuffle(data)
        
        self.X = data
        self.training_set_size = data.shape[0]

    def get_batch(self, batch_size, use_noise=False):
        indexes = np.random.randint(self.X.shape[0], size=batch_size)
        if use_noise:
            return self.X[indexes,:], add_gaussian_noise(self.X[indexes,:])
        
        # Assuming you want to resize images in the get_batch method
        resized_images = tf.image.resize(self.X[indexes,:], size=(8, 8), method=tf.image.ResizeMethod.BICUBIC)
        return self.X[indexes,:], resized_images
