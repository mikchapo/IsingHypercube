## This file contains code for reading the csv's output by the
## IsingMonteCarlo script. It is used for importing data into
## networks.py and compress_data.py.

from random import shuffle  # Randomizes list orders
from csv import reader  # Reads csv's
from os import listdir  # Lists all files in directory


## Takes a csv file for the Toric Code and reads the relevant
## information. The information for one laticce is stored in a tuple
## where the first element is a list containing the spins; the second is
## a list containing the phase information, where the first element
## refers to the low temperature phase, and the second element refers to
## the high temperature phase. The third element is the temperature of
## the lattice.
def unpackFile(file):
    data = ([], [], 0.0)    # Initializes data tuple, following the format
                            # explained above

    for i,row in enumerate(csv.reader(file)):
        if i == 0:
            if abs(float(row[7])) > 0.5:
                data[1] = [1.0,0.0]
            
            else:
                data[1] = [0.0,1.0]           
            
            data[2] = float(row[0])
        
        else:
            data[0] = data[0] + map(int,row)

    return data


def importLattices(directory, test_only=False, exclude=None,
                   exclude_range=0.0, training_limit=None):
    test_set = []
    test_files = os.listdir(directory + 'test/')
    counter = 0
    
    for test_file in test_files:
        counter += 1
        
        if (counter % 1000) == 0:
            print("Test CSV " + str(counter))
        
        file = open(directory + 'test/' + test_file, 'r')
        data = unpackFile(file)
        test_set.append(data)

    if not test_only:
        training_set = []
        training_files = os.listdir(directory + 'training/')
        counter = 0
        
        if training_limit == None:
            training_limit = len(training_files)
        
        for training_file in training_files:
            if counter < training_limit:
                counter += 1
                
                if (counter % 1000) == 0:
                    print("Training CSV " + str(counter))
                
                file = open(directory + 'training/' + training_file, 'r')
                data = unpackFile(file)

                # e_counter = 0
                # if exclude != None and (temp - (exclude - exclude_range)) > 0.0 and (temp - (exclude + exclude_range)) < 0.0:
                #     e_counter += 1
                #     if e_counter % 100 == 0:
                #         print "Excluded " + e_counter
                # else:
                #     training_set.append(data)

                training_set.append(data)
            
        rn.shuffle(training_set)

        validation_set = []
        validation_files = os.listdir(directory + 'validation/')

        counter = 0
        for validation_file in validation_files:
            counter += 1
            if (counter % 1000) == 0:
                print("Validation CSV " + str(counter))
            file = open(directory + 'validation/' + validation_file, 'r')
            data = unpackFile(file)
            validation_set.append(data)
            
        rn.shuffle(validation_set)

        return training_set, validation_set, test_set

    return test_set