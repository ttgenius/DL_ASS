import gzip
import numpy as np


def get_data(inputs_file_path, labels_file_path, num_examples):
    """
    Takes in an inputs file path and labels file path, unzips both files,
    normalizes the inputs, and returns (NumPy array of inputs, NumPy
    array of labels). Read the data of the file into a buffer and use
    np.frombuffer to turn the data into a NumPy array. Keep in mind that
    each file has a header of a certain size. This method should be called
    within the main function of the assignment.py file to get BOTH the train and
    test data. If you change this method and/or write up separate methods for
    both train and test data, we will deduct points.

    Hint: look at the writeup for sample code on using the gzip library

    :param inputs_file_path: file path for inputs, something like
    'MNIST_data/t10k-images-idx3-ubyte.gz'
    :param labels_file_path: file path for labels, something like
    'MNIST_data/t10k-labels-idx1-ubyte.gz'
    :param num_examples: used to read from the bytestream into a buffer. Rather
    than hardcoding a number to read from the bytestream, keep in mind that each image
    (example) is 28 * 28, with a header of a certain number.
    :return: NumPy array of inputs as float32 and labels as int8
    """

    # TODO: Load inputs and labels
    #read image file
    with gzip.open(inputs_file_path, 'rb') as f:
        image_buff = f.read()
        image_array = np.frombuffer(image_buff, dtype=np.uint8, offset=16)
        # reshape (ne * 784,) -> (ne, 784)
        image_array = image_array.reshape(num_examples, 784)
        # TODO: Normalize inputs
        # Normalize image array
        image_array = image_array.astype(np.float32) / 255

    #read lable file
    with gzip.open(labels_file_path, 'rb') as f:
        label_buff = f.read()
        label_array = np.frombuffer(label_buff, dtype=np.uint8, offset=8)

    return image_array, label_array


# i,l = get_data("C:/Users/zhang/00_uni/COSC440_Assignment1/COSC440_Assignment1/MNIST_data/t10k-images-idx3-ubyte.gz",
#         "C:/Users/zhang/00_uni/COSC440_Assignment1/COSC440_Assignment1/MNIST_data/t10k-labels-idx1-ubyte.gz", 10000)
# print("i",i.shape,i)#(10000,784)
# print("l",l.shape,l)#(10000,)
# num_classes = 10
# input_size = 784
# w = np.zeros((input_size, num_classes))  # (784, 10)
# b = np.zeros((1, num_classes))  # (1, 10)
# print("Dot",np.dot(i,w).shape)
# f=np.dot(i, w) + b  #(10000, 10)
# print("f",f.shape,f)
# batch_size = 100
# y=np.eye(batch_size) #(10,10)
# y = y[l, 0:num_classes]
# print("y",y.shape,y)  #(10000,10)
# predict_indexes = np.argmax(f, axis=1)
# predict_vals = np.eye(batch_size)[predict_indexes, 0:num_classes]
# print("predictvalues",predict_vals.shape,predict_vals)
# print("pindesx",predict_indexes.shape,predict_indexes)  #(10000,)
# err = y-predict_vals
# print("sss",sum(y-f))
# print("Err",err.shape, err,np.sum(err,axis=0))
# L = i @ w + b
# # Raise to e to the L for softmax function
# exp_L = np.exp(L)
# # Calculate sum for probability
# sum_L = np.sum(exp_L, 1)
# # Reshape to 2D matrix for matrix division
# sum_L = sum_L.reshape(sum_L.shape[0], 1)
# # Calculate the probability matrix
# probabilities = exp_L / sum_L
# print(probabilities.shape,probabilities)
# print("Test pro",np.sum(y - probabilities))
# out = np.matmul(i, w) + b
# prediction = np.argmax(out, axis=1)
# print("out",out.shape,out)
# predicted_values = np.eye(num_classes)[prediction]
#
# expected_values = np.eye(num_classes)[np.ravel(l)]
# print("predisciton", prediction.shape,prediction)
# print("prev",predicted_values.shape, predicted_values)
# print("expv", expected_values.shape, expected_values)
# print("D",np.diag(f[:,l]))
# a=(predict_vals == predicted_values)
# print(a.all())# R = np.arange(0, batch_size)
# print(np.ones((len(i), 1)))
# bia = np.matmul(np.transpose(err), np.ones((len(i), 1)))
# print(bia,bia.shape)
# print("+",f+err.sum(axis=0))
# print("i,t",np.matmul(i.T, err).shape) #(10,784)
# print(i.T.dot(err).shape)
# err_sum = np.sum(err)
# print("errsum",err_sum)
# print("Errsum",err_sum.shape)
# print("Adf",np.sum(err),np.sum(err).shape,np.sum(y).shape)
# print(np.matmul(np.transpose(y), i))
#
# Y = np.eye(batch_size)[l, 0:num_classes]
# gradW = i.T @ (Y - probabilities) / batch_size  # (784,10)
# gradB =  np.sum(Y - probabilities) / batch_size  # (1,10)
# print(gradB.shape)
#
# loss = expected_values - predicted_values
# gradient_weight = np.matmul(i.T, loss) / batch_size
# gradient_bias = np.sum(err) / batch_size
# print("xx",err,gradient_weight,gradient_bias)
# print("R",R)
# one_hot_labels = np.zeros((batch_size, num_classes))
# one_hot_labels[R, l.ravel()] = 1
# print("one hot lables",one_hot_labels.shape,one_hot_labels)
# # get the argmax of the outputs and compare them to the new label bois
# index = np.argmax(f, axis=1)
# output_compare = np.zeros((batch_size, num_classes))
# output_compare[R, index] = 1
# print("output compare",output_compare.shape,output_compare)