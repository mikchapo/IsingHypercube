from networks import neuralNet, plot_results

import matplotlib.pyplot as plt

import os

def create_unique_dir(parent_dir, name=None):
	trial_number = 1
	created = False
	while not created:
	    try:
	        if name != None:
	        	directory = (parent_dir + '/' + name + '(' +
	        		str(trial_number) + ')/')
	        else:
	        	directory = parent_dir + '/' + str(trial_number) + '/'
	        os.makedirs(directory)
	        created = True
	    except(OSError):
	        trial_number += 1

	return directory

def plot_accuracy(batches, init_acc, final_acc, train_accs_list,
	val_accs_list, iter_lists, hyper_params):

    batch_list = range(0, batches+report_freq, report_freq)
    trunc_batches = batches[1:]

    hp = hyper_params
    if len(iter_lists) == 1:
	    title = str(hp['numb_conv1']) + '(' + str(hp['field_size1']) + ')' + '-' + str(hp['numb_conv2']) + '(' + str(hp['field_size2']) +
        ')' + '-' + str(hp['numb_relu1']) + '-' + str(hp['numb_relu2']) + '-SS' + str(hp['step_size']) + '-RP' + str(hp['reg_pa'])
        '\nTraining Accuracy vs. Batch'
        fig,ax = plt.subplots(2)
        ax[0].set_title(title)
        ax[1].set_title('Validation Accuracy vs. Batch')

        ax[0].plot(trunc_batches,train_accs_list[0], 'b-')
	    ax[1].plot(batches,val_accs_list[0], 'b-')

	    plt.savefig(directory + str(numb_conv1))

	elif len(iter_lists) == 2:

	else:

def multi_variable_loop(dim, L, iter_lists, n, test_set, hyper_params,
	training=False, training_set=[], validation_set=[], print_args=[]):
	
	if n > 0:
		args = [dim, L, iter_lists, n-1, test_set]
		kwargs = {'training': training, 'training_set': training_set,
			'validation_set': validation_set}
		
		for item in iter_lists[n]['values']:
			if iter_lists[n]['label'] == 'Step Size':
				hyper_params['step_size'] = item
			
			if iter_lists[n]['label'] == 'ReLu 1':
				hyper_params['numb_relu1'] = item
			
			if iter_lists[n]['label'] == 'ReLu 2':
				hyper_params['numb_relu2'] = item
			
			if iter_lists[n]['label'] == 'Conv 1':
				hyper_params['numb_conv1'] = item
			
			if iter_lists[n]['label'] == 'Conv 2':
				hyper_params['numb_conv2'] = item
			
			if iter_lists[n]['label'] == 'Field Size 1':
				hyper_params['field_size1'] = item
			
			if iter_lists[n]['label'] == 'Field Size 2':
				hyper_params['field_size2'] = item
			args.append(hyper_params)

			print_args.extend([iter_lists[n]['label'], item])
			kwargs['print_args'] = print_args
			
			multi_variable_loop(*args, **kwargs)

	elif training:
	    train_accs_list = []
	    val_accs_list = []
	    
	    for item in iter_lists[n]['values']:
			print_args.extend([iter_lists[n]['label'], item])
			print(*print_args)

			init_acc, final_acc, train_accs, val_accs, temps, low_points, high_points = neuralNet(
				dim, L, test_set, training_set=training_set,
				validation_set=validation_set, **hyper_params)

			plot_results(L, temps, low_points, high_points, iter_lists, hyper_params)

			train_accs_list.append(train_accs)
			val_accs_list.append(val_accs)

		plot_accuracy(batches, init_acc, final_acc, train_accs_list,
			val_accs_list, iter_lists, hyper_params)

	else:
		for item in iter_lists[n]['values']:
			print_args.extend([iter_lists[n]['label'], item])
			print(*print_args)

			init_acc, final_acc, temps, low_points, high_points = neuralNet(dim, L, test_set, **hyper_params)

			plot_results(L, temps, low_points, high_points)




def train_network(dim, L,
	numb_relu1=None, numb_relu2=None, numb_conv1=None, numb_conv2=None, field_size1=2, field_size2=2,
	step_size=0.1, batches=2000, report_freq=100, reg_param=None,
	load=False, load_name="", save=False, save_name=None, acc_name=None,
	step_sizes={'label':'Step Size', 'values':[]}, numb_relu1_list={'label':'ReLu 1', 'values':[]},
	numb_relu2_list={'label':'ReLu 2', 'values':[]}, numb_conv1_list={'label':'Conv 1', 'values':[]},
	numb_conv2_list={'label':'Conv 2', 'values':[]}, field_size1_list={'label':'Field Size 1', 'values':[]},
	field_size2_list={'label':'Field Size 2', 'values'[]}):
	
	plot_colours = ['b','c','g','y','r','m']
	iter_lists = [step_sizes, numb_relu1_list, numb_relu2_list, numb_conv1_list, numb_conv2_list,
					field_size1_list, field_size2_list]
	hyper_params = {'batches': batches, 'report_freq': report_freq, 'reg_param': reg_param}
	
	for iter_list in iter_lists:
		if not iter_list['values']:
			iter_lists.remove(iter_list)
			
			if iter_list['label'] == 'Step Size':
				hyper_params['step_size'] = step_size
			
			if iter_list['label'] == 'ReLu 1':
				hyper_params['numb_relu1'] = numb_relu1
			
			if iter_list['label'] == 'ReLu 2':
				hyper_params['numb_relu2'] = numb_relu2
			
			if iter_list['label'] == 'Conv 1':
				hyper_params['numb_conv1'] = numb_conv1
			
			if iter_list['label'] == 'Conv 2':
				hyper_params['numb_conv2'] = numb_conv2
			
			if iter_list['label'] == 'Field Size 1':
				hyper_params['field_size1'] = field_size1
			
			if iter_list['label'] == 'Field Size 2':
				hyper_params['field_size2'] = field_size2
	
	if load:
		if L > 9:
		    model_path = "network_data/TC-D" + str(dim) + "-L" + str(L) + "/" + load_name + "model.ckpt"			
		else:
		    model_path = "network_data/TC-D" + str(dim) + "-L0" + str(L) + "/" + load_name + "model.ckpt"

		if isfile(model_path):
		    if L > 9:
		        test_set = importer('TC-D' + str(dim) + '-L' + str(L) + '/', test_only=True)
		    else:
		        test_set = importer('TC-D' + str(dim) + '-L0' + str(L) + '/', test_only=True)
		    
		    multi_variable_loop(dim, L, iter_lists, len(iter_lists), test_set, hyper_params)

    if L > 9:
        training_set, validation_set, test_set = importer('TC-D' + str(dim) + '-L' + str(L) + '/')
    else:
        training_set, validation_set, test_set = importer('TC-D' + str(dim) + '-L0' + str(L) + '/')

    multi_variable_loop(dim, L, iter_lists, len(iter_lists), test_set, hyper_params,
    	training=True, training_set=training_set, validation_set=validation_set)


L = 8
numb_layer1 = None
numb_layer2 = None
numb_conv1 = 4
numb_conv2 = None
field_size = 2
step_size = 0.075
number_of_batches = 3000
report_freq = 50
reg_param = 1