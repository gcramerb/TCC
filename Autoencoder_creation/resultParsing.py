import os
import json
import numpy as np
path = "../../resultados"
dir_list = os.listdir(path)
result_list = []
best_result = dict()
best_result['0.3'] = []
best_result['0.5'] = []
best_result['0.9'] = []
dict_file = dict()
min1 = 99
min2 = 99
min3 = 99
for file in dir_list:
	path_file = path + '/'+ file + '/informations.json'
	f = open(path_file,'r')
	j = json.load(f)
	error = (j['resultado']['RMSE']['autoEncoder  x'],j['resultado']['RMSE']['autoEncoder  y'],j['resultado']['RMSE']['autoEncoder  z'])
	m = j['missing']
	dict_file[file] = j
	if np.sum(error) < min1 and m == '0.3':
		min1 = np.sum(error)
		arq1 = file

	if np.sum(error) < min2 and m == '0.5':
		min2 = np.sum(error)
		arq2 = file
	if np.sum(error) < min3 and m == '0.9':
		min3 = np.sum(error)
		arq3 = file


try:
	print(arq1,arq2,arq3)
except:
	print('errorr')