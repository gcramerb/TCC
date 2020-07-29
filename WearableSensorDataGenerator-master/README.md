# Wearable Sensor Data Generator

![GitHub](https://img.shields.io/github/license/jessicasena/WearableSensorDataGenerator)

The `WearableSensorDataGenerator` is a tool to upgrade the very useful benchmark proposed by Artur et al. in the work *Human Activity Recognition Based on Wearable Sensor Data: A Standardization of the State-of-the-Art * ([GitHub repository](https://github.com/arturjordao/WearableSensorData "GitHub repository")) to a format that enables the use of Keras data generator. 

Currently, each dataset of the benchmark is available in a .npz file.  A drawback of this, however, is that we have to move the entire dataset to memory, which in some cases can prevent the use. To cope with that, we created a simple script that extracts the samples to a folder and creates files to maintain information regarding the sample labels and the division of each cross-validation fold (this information is important to ensure the maintenance of the standardization proposed by Artur). We also provide the data generator class that uses our extracted data e feeds the train_on_batch() or the fit.predict() Keras's functions, as well as an example of its usage.


## Requirements

- [Scikit-learn](http://scikit-learn.org/stable/)
- [Scipy](https://www.scipy.org/)
- [Keras](https://github.com/fchollet/keras) (Recommended version 2.2.0)
- [Tensorflow](https://www.tensorflow.org/) (Recommended version 1.5.0)
- [Python 3](https://www.python.org/)

## Quick Start
1. Clone this repository
2. Run
    ```bash
    python npz_to_fold.py -i <your/input/folder/> -o <your/output/folder/> -d <dataset1_name dataset2_name>
    ```
	For example
	```bash
   python npz_to_fold.py -i Z:/Datasets/ -o Z:/NewDatasets/ -d UTD-MHAD1_1s UTD-MHAD2_1s WHARF
   ```
3. And, enjoy it!

## Example of Use

Please refer to [example.py](https://github.com/jessicasena/WearableSensorDataGenerator/blob/master/example.py "example.py").


## License
See [LICENSE](https://github.com/jessicasena/WearableSensorDataGenerator/blob/master/LICENSE).

Please cite our paper in your publications if it helps your research.
```bash
@article{Jordao:2018,
author    = {Artur Jordao,
Antonio Carlos Nazare,
Jessica Sena and
William Robson Schwartz},
title     = {Human Activity Recognition Based on Wearable Sensor Data: A Standardization of the State-of-the-Art},
journal   = {arXiv},
year      = {2018},
eprint    = {1806.05226},
}
```
