import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
import pickle
import utils

# load datasets
test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')
train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')

# scale to [-1.0,1.0]
scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
test_images = scaler.fit_transform(test_images)
train_images = scaler.fit_transform(train_images)

# validation dataset
X = train_images
y = train_labels

sss = StratifiedShuffleSplit(test_size=0.1, random_state=1)
sss.get_n_splits(X, y)

for train_index, test_index in sss.split(X, y):
    train_images_2, validation_images = X[train_index], X[test_index]
    train_labels_2, validation_labels = y[train_index], y[test_index]

# resave the scaled and splitted numpy arrays
np.save('train_images_2.npy', train_images_2)
np.save('train_labels_2.npy', train_labels_2)
np.save('validation_images.npy', validation_images)
np.save('validation_labels.npy', validation_labels)
np.save('test_images_2.npy', test_images)


# part_3
# Classifier

training_loss_7 = np.empty([1, 100])
weight_first_7 = np.empty([784, 16])
weight_second_7 = np.empty([784, 16])
gradient_loss_7 = np.empty([1, 100])


mlp = MLPClassifier(hidden_layer_sizes=(16, 32, 64, 128, 64, 32, 16), alpha=0, activation='relu',
                    batch_size=500, learning_rate_init=0.01, momentum=0, )

for j in range(0, 100):
    for i in range(0, 10):
        mlp.partial_fit(train_images_2, train_labels_2, np.unique(train_labels_2))
        training_loss_7[:, j] = mlp.loss_
        weight_first_7 = mlp.coefs_[0]
        weight_second_7 = np.subtract(weight_second_7, weight_first_7)
    gradient_loss_7[:, j] = np.linalg.norm(weight_second_7)
    weight_second_7 = np.zeros([784, 16])
    train_images_2, train_labels_2 = shuffle(train_images_2, train_labels_2)
    print('9', j)

np.save('training_loss_7.npy', training_loss_7)
np.save('weight_first_7.npy', weight_first_7)
np.save('weight_second_7.npy', weight_second_7)
np.save('gradient_loss_7.npy', gradient_loss_7)


training_loss_l_7 = np.empty([1, 100])
weight_first_l_7 = np.empty([784, 16])
weight_second_l_7 = np.empty([784, 16])
gradient_loss_l_7 = np.empty([1, 100])


mlp = MLPClassifier(hidden_layer_sizes=(16, 32, 64, 128, 64, 32, 16), alpha=0, activation='logistic',
                    batch_size=500, learning_rate_init=0.01, momentum=0, )

for j in range(0, 100):
    for i in range(0, 10):
        mlp.partial_fit(train_images_2, train_labels_2, np.unique(train_labels_2))
        training_loss_l_7[:, j] = mlp.loss_
        weight_first_l_7 = mlp.coefs_[0]
        weight_second_l_7 = np.subtract(weight_second_l_7, weight_first_l_7)
    gradient_loss_l_7[:, j] = np.linalg.norm(weight_second_l_7)
    weight_second_l_7 = np.zeros([784, 16])
    train_images_2, train_labels_2 = shuffle(train_images_2, train_labels_2)
    print('10', j)

np.save('training_loss_l_7.npy', training_loss_l_7)
np.save('weight_first_l_7.npy', weight_first_l_7)
np.save('weight_second_l_7.npy', weight_second_l_7)
np.save('gradient_loss_l_7.npy', gradient_loss_l_7)

training_loss = np.load('training_loss.npy')
gradient_loss = np.load('gradient_loss.npy')
training_loss_sig = np.load('training_loss_sig.npy')
gradient_loss_sig = np.load('gradient_loss_l.npy')
training_loss_2 = np.load('training_loss_2.npy')
gradient_loss_2 = np.load('gradient_loss_2.npy')
training_loss_l_2  = np.load('training_loss_l_2.npy')
gradient_loss_l_2 = np.load('gradient_loss_l_2.npy')
training_loss_3 = np.load('training_loss_3.npy')
gradient_loss_3 = np.load('gradient_loss_3.npy')
training_loss_l_3 = np.load('training_loss_l_3.npy')
gradient_loss_l_3 = np.load('gradient_loss_l_3.npy')
training_loss_5 = np.load('training_loss_5.npy')
gradient_loss_5 = np.load('gradient_loss_5.npy')
training_loss_l_5 = np.load('training_loss_l_5.npy')
gradient_loss_l_5 = np.load('gradient_loss_l_5.npy')
training_loss_7 = np.load('training_loss_7.npy')
gradient_loss_7 = np.load('gradient_loss_7.npy')
training_loss_l_7 = np.load('training_loss_l_7.npy')
gradient_loss_l_7 = np.load('gradient_loss_l_7.npy')

training_loss = training_loss.reshape((-1, 1))
np.transpose(training_loss)
training_loss_sig = training_loss_sig.reshape((-1, 1))
np.transpose(training_loss_sig)
training_loss_2 = training_loss_2.reshape((-1, 1))
np.transpose(training_loss_2)
training_loss_l_2 = training_loss_l_2.reshape((-1, 1))
np.transpose(training_loss_l_2)
training_loss_3 = training_loss_3.reshape((-1, 1))
np.transpose(training_loss_3)
training_loss_l_3 = training_loss_l_3.reshape((-1, 1))
np.transpose(training_loss_l_3)
training_loss_5 = training_loss_5.reshape((-1, 1))
np.transpose(training_loss_5)
training_loss_l_5 = training_loss_l_5.reshape((-1, 1))
np.transpose(training_loss_l_5)
training_loss_7 = training_loss_7.reshape((-1, 1))
np.transpose(training_loss_7)
training_loss_l_7 = training_loss_l_7.reshape((-1, 1))
np.transpose(training_loss_l_7)
gradient_loss = gradient_loss.reshape((-1, 1))
np.transpose(gradient_loss)
gradient_loss_sig = gradient_loss_sig.reshape((-1, 1))
np.transpose(gradient_loss_sig)
gradient_loss_2 = gradient_loss_2.reshape((-1, 1))
np.transpose(gradient_loss_2)
gradient_loss_l_2 = gradient_loss_l_2.reshape((-1, 1))
np.transpose(gradient_loss_l_2)
gradient_loss_3 = gradient_loss_3.reshape((-1, 1))
np.transpose(gradient_loss_3)
gradient_loss_l_3 = gradient_loss_l_3.reshape((-1, 1))
np.transpose(gradient_loss_l_3)
gradient_loss_5 = gradient_loss_5.reshape((-1, 1))
np.transpose(gradient_loss_5)
gradient_loss_l_5 = gradient_loss_l_5.reshape((-1, 1))
np.transpose(gradient_loss_l_5)
gradient_loss_7 = gradient_loss_7.reshape((-1, 1))
np.transpose(gradient_loss_7)
gradient_loss_l_7 = gradient_loss_l_7.reshape((-1, 1))
np.transpose(gradient_loss_l_7)

dictionary_obj1_part3 = {'name': 'arch_1',
                         'relu_loss_curve': training_loss,
                         'sigmoid_loss_curve': training_loss_sig,
                         'relu_grad_curve': gradient_loss,
                         'sigmoid_grad_curve': gradient_loss_sig}

pickle.dump(dictionary_obj1_part3, open("part3_arch_1.p", "wb"))
result_1 = dictionary_obj1_part3

dictionary_obj2_part3 = {'name': 'arch_2',
                         'relu_loss_curve': training_loss_2,
                         'sigmoid_loss_curve': training_loss_l_2,
                         'relu_grad_curve': gradient_loss_2,
                         'sigmoid_grad_curve': gradient_loss_l_2}

pickle.dump(dictionary_obj2_part3, open("part3_arch_2.p", "wb"))
result_2 = dictionary_obj2_part3

dictionary_obj3_part3 = {'name': 'arch_3',
                         'relu_loss_curve': training_loss_3,
                         'sigmoid_loss_curve': training_loss_l_3,
                         'relu_grad_curve': gradient_loss_3,
                         'sigmoid_grad_curve': gradient_loss_l_3}

pickle.dump(dictionary_obj3_part3, open("part3_arch_3.p", "wb"))
result_3 = dictionary_obj3_part3

dictionary_obj4_part3 = {'name': 'arch_5',
                         'relu_loss_curve': training_loss_5,
                         'sigmoid_loss_curve': training_loss_l_5,
                         'relu_grad_curve': gradient_loss_5,
                         'sigmoid_grad_curve': gradient_loss_l_5}

pickle.dump(dictionary_obj4_part3, open("part3_arch_5.p", "wb"))
result_4 = dictionary_obj4_part3

dictionary_obj5_part3 = {'name': 'arch_7',
                         'relu_loss_curve': training_loss_7,
                         'sigmoid_loss_curve': training_loss_l_7,
                         'relu_grad_curve': gradient_loss_7,
                         'sigmoid_grad_curve': gradient_loss_l_7}

pickle.dump(dictionary_obj5_part3, open("part3_arch_7.p", "wb"))
result_5 = dictionary_obj5_part3

results = [result_1, result_2, result_3, result_4, result_5]
utils.part3Plots(results, save_dir='C:/Users/Ferdi/Desktop/EE496/Homework_1/e2093177_hw1', filename='part3Plots')