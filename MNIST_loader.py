#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#

import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
import torch
import random


# Set the seed for reproducibility
def set_seed(seed):

    if seed == None:
        seed = random.randint(0, 999999)
        print(f"(No seed given, using {seed} instead!)")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loadDataWrapper(seed=None):

    set_seed(seed)
    
    class MnistDataloader(object):
        def __init__(self, training_images_filepath,training_labels_filepath,
                     test_images_filepath, test_labels_filepath):
            self.training_images_filepath = training_images_filepath
            self.training_labels_filepath = training_labels_filepath
            self.test_images_filepath = test_images_filepath
            self.test_labels_filepath = test_labels_filepath
        
        def read_images_labels(self, images_filepath, labels_filepath):        
            labels = []
            with open(labels_filepath, 'rb') as file:
                magic, size = struct.unpack(">II", file.read(8))
                if magic != 2049:
                    raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
                labels = array("B", file.read())        
            
            with open(images_filepath, 'rb') as file:
                magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
                if magic != 2051:
                    raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
                image_data = array("B", file.read())        
            images = []
            for i in range(size):
                images.append([0] * rows * cols)
            for i in range(size):
                img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
                img = img.reshape(28, 28)
                images[i][:] = img            
            
            return images, labels
                
        def load_data(self):
            x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
            x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
            return (x_train, y_train),(x_test, y_test)        
    
    #
    # Set file paths based on added MNIST Datasets
    #
    input_path = ''#'../input'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    
    #
    # Load MINST dataset
    #
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # Convert to tensors
    x_train_array = np.array(x_train)
    x_train_tensor = torch.tensor(x_train_array)
    
    y_train_array = np.array(y_train)
    y_train_tensor = torch.tensor(y_train_array)
    
    x_test_array = np.array(x_test)
    x_test_tensor = torch.tensor(x_test_array)
    
    y_test_array = np.array(y_test)
    y_test_tensor = torch.tensor(y_test_array)
    

    return (x_train_tensor, y_train_tensor), (x_test_tensor, y_test_tensor)

