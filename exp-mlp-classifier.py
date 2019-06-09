import numpy as np
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


# part_2_1
# Classifier

# array declarations for saving outputs
train_accuracy = np.empty([10, 100])
validation_accuracy = np.empty([10, 100])
training_loss = np.empty([10, 100])
test_accuracy = np.empty([10, 1])
weight_first = np.empty([10, 784, 16])

# training and fitting data
for k in range(0, 10):
    mlp = MLPClassifier(hidden_layer_sizes=(16, 128, 64, 32, 16), alpha=0,
                        batch_size=500, max_iter=100)
    for j in range(0, 100):
        for i in range(0, 10):
            mlp.partial_fit(train_images_2, train_labels_2, np.unique(train_labels_2))
            train_accuracy[k, j] = mlp.score(train_images_2, train_labels_2)
            validation_accuracy[k, j] = mlp.score(validation_images, validation_labels)
            training_loss[k, j] = mlp.loss_
        train_images_2, train_labels_2 = shuffle(train_images_2, train_labels_2)
    test_accuracy[k] = mlp.score(test_images, test_labels)
    weight_first[k] = mlp.coefs_[0]
    print(k)

# saving results into numpy arrays

np.save('train_accuracy_5.npy', train_accuracy)
np.save('validation_accuracy_5.npy', validation_accuracy)
np.save('training_loss_5.npy', training_loss)
np.save('test_accuracy_5.npy', test_accuracy)
np.save('weight_first_5.npy', weight_first)

# load saved arrays

test_accuracy = np.load('test_accuracy.npy')
train_accuracy = np.load('train_accuracy.npy')
validation_accuracy = np.load('validation_accuracy.npy')
training_loss = np.load('training_loss.npy')
weight_first = np.load('weight_first.npy')

# finding results
best_test_accuracy = max(test_accuracy)
a, b = np.where(test_accuracy == test_accuracy.max())
average_train_accuracy = np.mean(train_accuracy, axis=0)
average_validation_accuracy = np.mean(validation_accuracy, axis=0)
average_training_loss = np.mean(training_loss, axis=0)
best_weight = weight_first[a]

# creating the dictionary object
dictionary_obj = {'name': 'arch_1',
                  'loss_curve': average_training_loss,
                  'train_acc_curve': average_train_accuracy,
                  'val_acc_curve': average_validation_accuracy,
                  'test_acc': best_test_accuracy,
                  'weights': best_weight}

pickle.dump(dictionary_obj, open("part2_arch_1.p", "wb"))

# reading results and forming the results list
result_1 = pickle.load(open("part2_arch_1.p", "rb"))
result_2 = pickle.load(open("part2_arch_2.p", "rb"))
result_3 = pickle.load(open("part2_arch_3.p", "rb"))
result_4 = pickle.load(open("part2_arch_5.p", "rb"))
result_5 = pickle.load(open("part2_arch_7.p", "rb"))

results = [result_1, result_2, result_3, result_4, result_5]

# plotting weights and curves for part 2
weights = best_weight

utils.part2Plots(results, save_dir='C:/Users/Toshiba-PC/Desktop/e2093177_hw1', filename='part2Plots')
utils.visualizeWeights(weights, save_dir='C:/Users/Toshiba-PC/Desktop/e2093177_hw1', filename='input_weights_1')
