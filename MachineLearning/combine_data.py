import os

def combineData(directory, data_filename):
    filenames = os.listdir(directory)
    filenames.sort()
    data_file = open(data_filename,'r')
    data = data_file.readlines()    

    for i in range(len(filenames)):
    	file = open(directory + filenames[i],'r+')
    	content = file.read()
        refined_data = data[i].replace(' ', ',')
        file.seek(0, 0)
        file.write(refined_data.rstrip('\r\n') + '\n' + content)

def combineDirectories(real_dir, obs_dir):
    filenames = os.listdir(obs_dir)
    if 'test' in filenames:
        filenames.remove('test')
    if 'training' in filenames:
        filenames.remove('training')
    if 'validation' in filenames:
        filenames.remove('validation')
    if 'measurements' in filenames:
        filenames.remove('measurements')
    for filename in filenames:
        os.rename(obs_dir + filename, real_dir + filename)

def removeData(start_dir, end_dir, temperature):
    filenames = os.listdir(start_dir)
    if 'test' in filenames:
        filenames.remove('test')
    if 'training' in filenames:
        filenames.remove('training')
    if 'validation' in filenames:
        filenames.remove('validation')
    if 'measurements' in filenames:
        filenames.remove('measurements')
    for filename in filenames:
        data_file = open(start_dir + filename)
        data = data_file.readline()
        data = data.split(",")
        T = float(data[0])
        if T > (temperature - 0.0001) and T < (temperature + 0.0001):
            os.rename(start_dir + filename, end_dir + filename)

combineDirectories('data/TC-D3-L08/','data/TC-D3-L08-Raw')
#removeData('TC-D2-L16/', 'TC-D2-L16-Unused-Sets/TC-D2-L16-T100/', 100.0)