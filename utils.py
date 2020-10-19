import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'C:\\Users\gcram\Documents\GitHub\TCC\TCC\missing generator\\')
import missing_creator as mc


def target_names(dataset_name):
	class_names = ""
	if dataset_name == 'MHEALTH':
		actNameMHEALTH = {
            0: 'Standing still',
            1: 'Sitting and relaxing',
            2: 'Lying down',
            3: 'Walking',
            4: 'Climbing stairs',
            5: 'Waist bends forward',
            6: 'Frontal elevation\nof arms',
            7: 'Knees bending\n(crouching)',
            8: 'Cycling',
            9: 'Jogging',
            10: 'Running',
            11: 'Jump front and back'
        }
		class_names = actNameMHEALTH

	elif dataset_name == 'PAMAP2P':

		actNamePAMAP2P = {
			0: 'lying',
			1: 'sitting',
			2: 'standing',
			3: 'ironing',
			4: 'vacuum cleaning',
			5: 'ascending stairs',
			6: 'descending stairs',
			7: 'walking',
			8: 'Nordic walking',
			9: 'cycling',
			10: 'running',
			11: 'rope jumping', }
		actNamePAMAP2P_v2 = {
			0: 'Lie',
			1: 'Sit',
			2: 'Stand',
			3: 'Iron',
			4: 'Break',
			5: 'Ascend stairs',
			6: 'Nordic walking',
			7: 'watching TV',
			8: 'computer work',
			9: 'car driving',
			10: 'ascending stairs',
			11: 'descending stairs',
			12: 'vacuum cleaning',
			13: 'ironing',
			14: 'folding laundry',
			15: 'house cleaning',
			16: 'playing soccer',
			17: 'rope jumping',
			18: 'other'}
		class_names = actNamePAMAP2P
	elif dataset_name == 'UTD-MHAD1_1s':

		actNameUTDMHAD = {
			0: 'right arm swipe\nto the left',
			1: 'right arm swipe\nto the right',
			2: 'right hand\nwave',
			3: 'two hand\nfront clap',
			4: 'right arm throw',
			5: 'cross arms\nin the chest',
			6: 'basketball shooting',
			7: 'draw x',
			8: 'draw circle\nclockwise',
			9: 'draw circle\ncounter clockwise',
			10: 'draw triangle',
			11: 'bowling',
			12: 'front boxing',
			13: 'baseball swing\nfrom right',
			14: 'tennis forehand\nswing',
			15: 'arm curl',
			16: 'tennis serve',
			17: 'two hand push',
			18: 'knock on door',
			19: 'hand catch',
			20: 'pick up\nand throw'
		}
		class_names = actNameUTDMHAD
	elif dataset_name == 'UTD-MHAD2_1s':

		actNameUTDMHAD2 = {
			0: 'jogging',
			1: 'walking',
			2: 'sit to stand',
			3: 'stand to sit',
			4: 'forward lunge',
			5: 'squat'}
		class_names = actNameUTDMHAD2

	elif dataset_name == 'WHARF':

		actNameWHARF = {

			0: 'Standup chair',
			1: 'Comb hair',
			2: 'Sitdown chair',
			3: 'Walk',
			4: 'Pour water',
			5: 'Drink glass',
			6: 'Descend stairs',
			7: 'Climb stairs',
			8: 'Liedown bed',
			9: 'Getup bed',
			10: 'Use telephone',
			11: 'Brush teeth'}
		class_names = actNameWHARF

	elif dataset_name == 'USCHAD':

		actNameUSCHAD = {
			0: 'Walking Forward',
			1: 'Walking Left',
			2: 'Walking Right',
			3: 'Walking Upstairs',
			4: 'Walking Downstairs',
			5: 'Running Forward',
			6: 'Jumping Up',
			7: 'Sitting',
			8: 'Standing',
			9: 'Sleeping',
			10: 'Elevator Up',
			11: 'Elevator Down'}
		class_names = actNameUSCHAD
	elif dataset_name == 'WISDM':

		actNameWISDM = {
			0: 'Jogging',
			1: 'Walking',
			2: 'Upstairs',
			3: 'Downstairs',
			4: 'Sitting',
			5: 'Standing'
		}
		class_names = actNameWISDM
	return class_names

def get_data(dataset_name):

    data_input_file = 'C:\\Users\gcram\Documents\Smart Sense\Datasets\\' + dataset_name + '.npz'
    label_names = target_names(dataset_name)
    batch = 128
    sensor_factor = '1.1.0'
    X, y, folds = mc.load_data(data_input_file=data_input_file,sensor_factor = sensor_factor)
    return X,y,folds,label_names


def plot_sensor(true,pred,file_name,label):
	sensor = ['acc','gyr','mag']
	axis = [' x',' y',' z']
	df_true = pd.DataFrame()
	df_pred = pd.DataFrame()
	for i in range(true.shape[-1]):
		col_true = sensor[int(i/3)] + axis[i%3] + ' true'
		pd.concat([df_true,pd.DataFrame(data = true[:,i],columns = [col_true])],axis = 1)
		col_pred = sensor[int(i/3)] + axis[i%3] + ' pred'
		pd.concat([df_pred,pd.DataFrame(data = pred[:,i],columns = [col_pred])],axis = 1)



	#fig = df.plot().get_figure()
	#fig.savefig('mhealth_reconstructed.png')
    #def plot_sensor(acc, acc_rec, label, model_name)
	f, axarr = plt.subplots(2, sharex=True, sharey=True)
    # pyplot.figure()

    # determine the total number of plots
    # n, off = imgs_B.shape[2] + 1, 0
    #sensor = np.squeeze(acc)
    # plot total acc
	axarr[0].plot(true[:, 0], color='green',label = 'x')
	axarr[0].plot(true[:, 1], color='blue',label = 'y')
	axarr[0].plot(true[:, 2], color='red',label = 'z')
	axarr[0].set_title('ACC Original - {}'.format(label))
	axarr[0].legend()
    # plot total acc
	axarr[1].plot(pred[:, 0], color='green',label = 'x')
	axarr[1].plot(pred[:, 1], color='blue',label = 'y')
	axarr[1].plot(pred[:, 2], color='red',label = 'z')
	axarr[1].set_title('ACC reconstructed')
	axarr[1].legend()

	plt.savefig("C:\\Users\gcram\Documents\Github\TCC\images\%s_%s.png" % (label, file_name))
	plt.close()
