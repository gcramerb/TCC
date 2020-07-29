# ---------------------------------------------------------------------------------------
# This code uncompress .npz data provided by the benchmark "Human Activity Recognition
# Based on Wearable Sensor Data: A Standardization of the State-of-the-Art"
# into a folder composed of individual samples and two files that holds information
# regarding the labels and on the separation of the folders used in the protocol.
#
# (C) 2020 Jéssica Sena, Brazil
# Released under GNU Public License (GPL)
# email jessicasenasouza@gmail.com
# ---------------------------------------------------------------------------------------
import numpy as np
import pickle
import os
import argparse


def npz_to_fold(input_folder, output_folder, dataset_name):
    """Uncompressed .npz data and save as individual samples"""
    data_output = os.path.join(output_folder, dataset_name)
    data_input_file = os.path.join(input_folder, dataset_name + ".npz")

    tmp = np.load(data_input_file, allow_pickle=True)
    X = tmp['X']
    y = tmp['y']
    fold_samples = []
    folds = tmp['folds']
    labels = {}

    print(dataset_name)
    print(X.shape)

    samples_output_folder = os.path.join(data_output, "samples/")
    if not os.path.exists(samples_output_folder):
        os.makedirs(samples_output_folder)

    for i in range(len(X)):
        id_sample = '{:06d}'.format(i)
        np.save(samples_output_folder + id_sample, X[i])
        labels[id_sample] = y[i]

    for i in range(len(folds)):
        fold_samples.append([[], []])
        for item in folds[i][0]:
            fold_samples[i][0].append('{:06d}'.format(item))
        for item in folds[i][1]:
            fold_samples[i][1].append('{:06d}'.format(item))

    np.save(data_output + "/folds", fold_samples)
    f = open(data_output + "/labels.pkl", "wb")
    pickle.dump(labels, f)
    f.close()
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", dest="input_folder", help="Benchmark's .npz files location", type=str,
                        required=True)
    parser.add_argument("-o", "--output_folder", dest="output_folder", help="Where to write the data", type=str,
                        required=True)
    parser.add_argument("-d", "--dataset_list", dest="dataset_list",
                        help="List of datasets to apply the transformation", nargs='+', required=True)

    args = parser.parse_args()

    for dataset in args.dataset_list:
        npz_to_fold(args.input_folder, args.output_folder, dataset)
