from os import listdir

def lattice_counter(dim, L):
	if L > 9:
		filenames = listdir("../data/TC-D%i-L%i-Raw/" % (dim, L))
	else:
		filenames = listdir("../data/TC-D%i-L0%i-Raw/" % (dim, L))

	temps = []
	for filename in filenames:
		new_temp = filename[1:6]
		# print repr(new_temp)		
		new_temp = new_temp.replace(",", ".")
		# print repr(new_temp)
		new_temp = float(new_temp)
		included = False
		for temp in temps:
			if temp['temp'] == new_temp:
				temp['count'] += 1
				included = True
				break
		if not included:
			temps.append({'temp': new_temp, 'count': 1})

	for temp in temps:
		print str(temp['temp']) + " " + str(temp["count"])

lattice_counter(3,12)
