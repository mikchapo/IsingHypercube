import random as rn
import numpy as np

import csv,os

def importLattices(directory, test_only=False, exclude=None, exclude_range=0.0, training_limit=None):
    test_set = []
    test_files = os.listdir(directory + 'test/')

    counter = 0
    for test_file in test_files:
        counter += 1
        if (counter % 1000) == 0:
            print("Test CSV " + str(counter))
        file = open(directory + 'test/' + test_file, 'r')
        data = {}
        data['values'] = []

        for i,row in enumerate(csv.reader(file)):
            if i == 0:
                if abs(float(row[7])) > 0.5:
                    data['phase'] = [1.0,0.0]
                else:
                    data['phase'] = [0.0,1.0]
                # data['wilx'] = [abs(float(row[7]))]
                
                data['T'] = float(row[0])
            else:
                data['values'] = data['values'] + map(int,row)

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
                data = {}
                data['values'] = []

                for i,row in enumerate(csv.reader(file)):
                    if i == 0:
                        if abs(float(row[7])) > 0.5:
                            data['phase'] = [1.0,0.0]
                        else:
                            data['phase'] = [0.0,1.0]
                        # data['wilx'] = [abs(float(row[7]))]

                        data['T'] = float(row[0])
                    else:
                        data['values'] = data['values'] + map(int,row)

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
            data = {}
            data['values'] = []

            for i,row in enumerate(csv.reader(file)):
                if i == 0:
                    if abs(float(row[7])) > 0.5:
                        data['phase'] = [1.0,0.0]
                    else:
                        data['phase'] = [0.0,1.0]
                    # data['wilx'] = [abs(float(row[7]))]
                    
                    data['T'] = float(row[0])
                else:
                    data['values'] = data['values'] + map(int,row)

            else:
                validation_set.append(data)
            
        rn.shuffle(validation_set)

        return training_set, validation_set, test_set

    return test_set