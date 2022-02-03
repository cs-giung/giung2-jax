import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


NAMES = np.array([
    "MNIST",
    "KMNIST",
    "FashionMNIST",
    "CIFAR10",
    "CIFAR100",
    # "TinyImageNet200",
])

EXISTS = np.array([os.path.exists(e) for e in NAMES])
NAMES = NAMES[EXISTS]

for name in NAMES:

    # load images and labels
    trn_images = np.load(f'{name}/train_images.npy')
    trn_labels = np.load(f'{name}/train_labels.npy')
    tst_images = np.load(f'{name}/test_images.npy')
    tst_labels = np.load(f'{name}/test_labels.npy')

    # just flatten images
    trn_images = trn_images.reshape(trn_images.shape[0], -1)
    tst_images = tst_images.reshape(tst_images.shape[0], -1)

    # run k-NN classifier
    knn = KNeighborsClassifier(
        n_neighbors = 3,
        weights     = 'distance',
        algorithm   = 'auto',
        n_jobs      = -1,
    )
    knn.fit(trn_images, trn_labels)
    print(f'{name} : ACC={knn.score(tst_images, tst_labels):.4f}')
