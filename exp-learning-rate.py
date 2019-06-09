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

# 'arch_7' is selected array declarations
training_loss_part4_1 = np.empty([1, 100])
validation_accuracy_part4_1 = np.empty([1, 100])

# classifier object created
mlp1 = MLPClassifier(hidden_layer_sizes=(16, 32, 64, 128, 64, 32, 16), alpha=0, solver='sgd',
                     batch_size=500, learning_rate_init=0.1, momentum=0, )

# for 100 epochs apply partial fit and score validation accuracy and training loss
for j in range(0, 100):
    for i in range(0, 10):
        mlp1.partial_fit(train_images_2, train_labels_2, np.unique(train_labels_2))
        validation_accuracy_part4_1[:, j] = mlp1.score(validation_images, validation_labels)
        training_loss_part4_1[:, j] = mlp1.loss_
    train_images_2, train_labels_2 = shuffle(train_images_2, train_labels_2)
    print('1', j)

# save the results
np.save('training_loss_part4_1.npy', training_loss_part4_1)
np.save('validation_accuracy_part4_1', validation_accuracy_part4_1)

training_loss_part4_2 = np.empty([1, 100])
validation_accuracy_part4_2 = np.empty([1, 100])

mlp2 = MLPClassifier(hidden_layer_sizes=(16, 32, 64, 128, 64, 32, 16), alpha=0, solver='sgd',
                     batch_size=500, learning_rate_init=0.01, momentum=0, )

for j in range(0, 100):
    for i in range(0, 10):
        mlp2.partial_fit(train_images_2, train_labels_2, np.unique(train_labels_2))
        validation_accuracy_part4_2[:, j] = mlp2.score(validation_images, validation_labels)
        training_loss_part4_2[:, j] = mlp2.loss_
    train_images_2, train_labels_2 = shuffle(train_images_2, train_labels_2)
    print('2', j)


np.save('training_loss_part4_2.npy', training_loss_part4_2)
np.save('validation_accuracy_part4_2', validation_accuracy_part4_2)

training_loss_part4_3 = np.empty([1, 100])
validation_accuracy_part4_3 = np.empty([1, 100])

mlp3 = MLPClassifier(hidden_layer_sizes=(16, 32, 64, 128, 64, 32, 16), alpha=0, solver='sgd',
                     batch_size=500, learning_rate_init=0.001, momentum=0, )

for j in range(0, 100):
    for i in range(0, 10):
        mlp3.partial_fit(train_images_2, train_labels_2, np.unique(train_labels_2))
        validation_accuracy_part4_3[:, j] = mlp3.score(validation_images, validation_labels)
        training_loss_part4_3[:, j] = mlp3.loss_
    train_images_2, train_labels_2 = shuffle(train_images_2, train_labels_2)
    print('3', j)


np.save('training_loss_part4_3.npy', training_loss_part4_3)
np.save('validation_accuracy_part4_3', validation_accuracy_part4_3)


# transpose the arrays (1-D arrays reshaped)
training_loss_part4_1 = training_loss_part4_1.reshape((-1, 1))
np.transpose(training_loss_part4_1)
training_loss_part4_2 = training_loss_part4_2.reshape((-1, 1))
np.transpose(training_loss_part4_2)
training_loss_part4_3 = training_loss_part4_3.reshape((-1, 1))
np.transpose(training_loss_part4_3)
validation_accuracy_part4_1 = validation_accuracy_part4_1.reshape((-1, 1))
np.transpose(validation_accuracy_part4_1)
validation_accuracy_part4_2 = validation_accuracy_part4_2.reshape((-1, 1))
np.transpose(validation_accuracy_part4_2)
validation_accuracy_part4_3 = validation_accuracy_part4_3.reshape((-1, 1))
np.transpose(validation_accuracy_part4_3)

# dictionary object
dictionary_obj_part4 = {'name': 'arch_7',
                        'loss_curve_1': training_loss_part4_1,
                        'loss_curve_01': training_loss_part4_2,
                        'loss_curve_001': training_loss_part4_3,
                        'val_acc_curve_1': validation_accuracy_part4_1,
                        'val_acc_curve_01': validation_accuracy_part4_2,
                        'val_acc_curve_001': validation_accuracy_part4_3}

# save the dictionary
pickle.dump(dictionary_obj_part4, open("part4_arch_7.p", "wb"))

# plot the curves for the part
result = dictionary_obj_part4
utils.part4Plots(result, save_dir='C:/Users/Ferdi/Desktop/EE496/Homework_1/e2093177_hw1', filename='part4Plots')

# epoch step found as 20 and parameter is adjusted accordingly
validation_accuracy_part4_5 = np.empty([1, 100])

mlp1 = MLPClassifier(hidden_layer_sizes=(16, 32, 64, 128, 64, 32, 16), alpha=0, solver='sgd',
                     batch_size=500, learning_rate_init=0.1, momentum=0, )

for j in range(0, 100):
    if j == 20:
        mlp1.set_params(learning_rate_init=0.01)
    for i in range(0, 10):
        mlp1.partial_fit(train_images_2, train_labels_2, np.unique(train_labels_2))
        validation_accuracy_part4_5[:, j] = mlp1.score(validation_images, validation_labels)
    train_images_2, train_labels_2 = shuffle(train_images_2, train_labels_2)
    print('1', j)

np.save('validation_accuracy_part4_5.npy', validation_accuracy_part4_5)

# epoch step is found as 40 and parameter is adjusted
validation_accuracy_part4_6 = np.empty([1, 100])

mlp1 = MLPClassifier(hidden_layer_sizes=(16, 32, 64, 128, 64, 32, 16), alpha=0, solver='sgd',
                     batch_size=500, learning_rate_init=0.1, momentum=0, )

for j in range(0, 100):
    if j == 20:
        mlp1.set_params(learning_rate_init=0.01)
    elif j == 40:
        mlp1.set_params(learning_rate_init=0.001)
    for i in range(0, 10):
        mlp1.partial_fit(train_images_2, train_labels_2, np.unique(train_labels_2))
        validation_accuracy_part4_6[:, j] = mlp1.score(validation_images, validation_labels)
    train_images_2, train_labels_2 = shuffle(train_images_2, train_labels_2)
    print('1', j)

np.save('validation_accuracy_part4_6.npy', validation_accuracy_part4_6)







