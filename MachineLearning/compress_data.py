from lattice_importer import importLattices
import gzip

try:
    import cPickle as pickle
except:
    import pickle

def compressData(dim, L):
	if L < 10:
		dir_name = "TC-D%i-L0%i" % (dim,L)
	else:
		dir_name = "TC-D%i-L%i" % (dim,L)

	directory = "../data/%s/" % dir_name
	
	data_sets = ([],[],[])
	data_sets = importLattices(directory)
	
	with gzip.open('../data/%s.pkl.gz' % dir_name, 'wb') as f:
		for data_set in data_sets:
			data_set = pickle.dumps(data_set)
    		f.write(data_set)

compressData(3,4)