import numpy as np
import struct 

def get_data():
    """
    Returns tuple (training_data, testing_data) where training_data is a list
    of tuples (training_image, training_label) where the training_label is
    the label corresponding to the image. testing_data is similar to
    training_data. 
    """

    ftrain_images = "data/train-images-idx3-ubyte"
    ftrain_labels = "data/train-labels-idx1-ubyte"
    ftest_images = "data/t10k-images-idx3-ubyte"
    ftest_labels = "data/t10k-labels-idx1-ubyte"

    with open(ftrain_images, "rb") as train_img:
        magic_num, train_num, rows, cols = struct.unpack(">iiii", train_img.read(16))
        training_images = np.fromfile(train_img,
                dtype=np.uint8).reshape(train_num, rows*cols,1)
    
    with open(ftrain_labels, "rb") as train_lbls:
        magic_num, train_num = struct.unpack(">ii", train_lbls.read(8))
        training_labels = np.fromfile(train_lbls, dtype=np.int8)

    with open(ftest_images, "rb") as test_img:
        magic_num, test_num, rows, cols = struct.unpack(">iiii", test_img.read(16))
        testing_images = np.fromfile(test_img,
                dtype=np.uint8).reshape(test_num, rows*cols,1)
        
    with open(ftest_labels, "rb") as test_lbls:
        magic_num, test_num = struct.unpack(">ii", test_lbls.read(8))
        testing_labels = np.fromfile(test_lbls, dtype=np.int8)

    pixel_normalization = 255
    training_images = training_images.astype(np.float32)
    training_images = training_images/pixel_normalization
    training_labels = [vectorize(training_labels[x]) for x in xrange(len(training_labels))]
    testing_images = testing_images.astype(np.float32)
    testing_images = testing_images/pixel_normalization
    testing_labels = [vectorize(testing_labels[x]) for x in xrange(len(testing_labels))]

    training_data = zip(training_images, training_labels)
    testing_data = zip(testing_images, testing_labels)
    return (training_data, testing_data)

def vectorize(y):
    """
    Vectorizes label data to account for NN design
    """
    expected_output = np.zeros([10,1])
    expected_output[y] = float(1)
    return expected_output
