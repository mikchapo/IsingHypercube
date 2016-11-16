import random as rn

import os

def assignLattices(directory, test_number, validation_number):
    filenames = os.listdir(directory)
    if 'test' in filenames:
        filenames.remove('test')
    if 'training' in filenames:
        filenames.remove('training')
    if 'validation' in filenames:
        filenames.remove('validation')
    if 'measurements' in filenames:
        filenames.remove('measurements')
    rn.shuffle(filenames)

    counter = 0
    for filename in filenames:
        if not os.path.exists(directory + 'test/'):
            os.makedirs(directory + 'test/')
        if not os.path.exists(directory + 'training/'):
            os.makedirs(directory + 'training/')
        if not os.path.exists(directory + 'validation/'):
            os.makedirs(directory + 'validation/')

        if counter < test_number:
            os.rename(directory + filename, directory + 'test/' + filename)
        elif counter < test_number + validation_number:
            os.rename(directory + filename, directory + 'validation/' + filename)
        else:
            try:
                os.rename(directory + filename, directory + 'training/' + filename)
            except(OSError):
                print directory + 'training/' + filename
        
        counter += 1
        if (counter % 1000) == 0:
            print("Moved " + str(counter))

def resetAssignment(directory):
    if not os.path.exists(directory + 'test/'):
        os.makedirs(directory + 'test/')
    if not os.path.exists(directory + 'training/'):
        os.makedirs(directory + 'training/')
    if not os.path.exists(directory + 'validation/'):
        os.makedirs(directory + 'validation/')

    test_files = os.listdir(directory + 'test/')
    if 'test' in test_files:
        test_files.remove('test')
    if 'training' in test_files:
        test_files.remove('training')
    if 'validation' in test_files:
        test_files.remove('validation')
    if 'measurements' in test_files:
        test_files.remove('measurements')
    for test_file in test_files:
        os.rename(directory + 'test/' + test_file, directory + test_file)

    training_files = os.listdir(directory + 'training/')
    if 'test' in training_files:
        training_files.remove('test')
    if 'training' in training_files:
        training_files.remove('training')
    if 'validation' in training_files:
        training_files.remove('validation')
    if 'measurements' in training_files:
        training_files.remove('measurements')
    for training_file in training_files:
        os.rename(directory + 'training/' + training_file, directory + training_file)

    validation_files = os.listdir(directory + 'validation/')
    if 'test' in validation_files:
        validation_files.remove('test')
    if 'training' in validation_files:
        validation_files.remove('training')
    if 'validation' in validation_files:
        validation_files.remove('validation')
    if 'measurements' in validation_files:
        validation_files.remove('measurements')
    for validation_file in validation_files:
        os.rename(directory + 'validation/' + validation_file, directory + validation_file)

resetAssignment('data/TC-D3-L08/')
assignLattices('data/TC-D3-L08/',1000,500)