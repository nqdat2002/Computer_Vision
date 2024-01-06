import os
import numpy as np
import shutil
import random

path = 'dataset/'
newPath = 'dataset_split/'

categories = os.listdir(path)
print(categories)

for category in categories:
    os.makedirs(newPath + 'train/' + category)
    os.makedirs(newPath + 'val/' + category)
    os.makedirs(newPath + 'test/' + category)

for category in categories:
    src = path + category  # folder to copy images from
    print(src)

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)

    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames) * 0.7),
                                                               int(len(allFileNames) * 0.9)])

    train_FileNames = [src + '/' + name for name in train_FileNames]
    val_FileNames = [src + '/' + name for name in val_FileNames]
    test_FileNames = [src + '/' + name for name in test_FileNames]

    print('Total images  : ' + category + ' ' + str(len(allFileNames)))
    print('Training : ' + category + ' ' + str(len(train_FileNames)))
    print('Validation : ' + category + ' ' + str(len(val_FileNames)))
    print('Testing : ' + category + ' ' + str(len(test_FileNames)))

    for name in train_FileNames:
        if name.endswith('.jpg') or name.endswith('.jpeg'):
            shutil.copy(name, newPath + 'train/' + category)

    for name in val_FileNames:
        if name.endswith('.jpg') or name.endswith('.jpeg'):
            shutil.copy(name, newPath + 'val/' + category)

    for name in test_FileNames:
        if name.endswith('.jpg') or name.endswith('.jpeg'):
            shutil.copy(name, newPath + 'test/' + category)

