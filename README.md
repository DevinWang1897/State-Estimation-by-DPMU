# State-estimation-by-DPMU
State estimation by DPMU

# Overview
This project uses a neural network model for state estimation of the distribution system. The input and output are as follows:
Input of the model: hour and hour data in cyclical encoding (cos,sin); weekday/weekend, holiday; voltage and current measurement of a DPMU.
Output of the model: the voltage magnitude of the feeder's smart meters.

Both the DPMU and smart meter data contains noise. However, when evaluating the testing performance, we should use noiseless smart meter data for reference.

# How to use
1. Before using the code, a user should prepare a folder containing all the data files needed to train the state estimation neural network model. In this folder, the following files should be prepared. Note that all the voltage magnitudes are in per unit, all the current magnitudes are in Ampere, and all the phase angles are in radian.
(1) uPMU_V_list.csv: This is a table containing all the potential nodes in the distribution network whose voltage can be measured by a DPMU, i.e., the DPMU voltage locations. Each node is represented by a number.
(2) Branch_list.csv: This is a table containing all the potential branches in the distribution network whose current can be measured by a DPMU, i.e., the DPMU current locations. Each branch is represented by two nodes and each node is represented by a number.
(3) uPMU_V_mag.csv: This is a table of DPMU voltage magnitude measurement of all the potential DPMU voltage locations. Each 3 columns represent the 3-phase voltage magnitude of a node.
(4) uPMU_V_rad.csv: This is a table of DPMU voltage angle measurement of all the potential DPMU voltage locations. Each 3 columns represent the 3-phase voltage angle of a node.
(5) uPMU_I_mag.csv: This is a table of DPMU current magnitude measurement of all the potential DPMU current locations. Each 3 columns represent the 3-phase current magnitude of a branch.
(6) uPMU_I_rad.csv: This is a table of DPMU current angle measurement of all the potential DPMU current locations. Each 3 columns represent the 3-phase current angle of a branch.
(7) datae_indicator.csv: This is a table showing the timestamps of the data. It contains 6 columns: year, month, day, hour, weekend (1 for weekends, 0 for weekdays), and holiday (1 for holidays, 0 for non-holidays). 
(8) smart_meter_volt.csv: The table of smart meter voltage measurement. Each column is a smart meter. This is the noisy smart meter data.
(9) smart_meter_volt_noiseless.csv: The table of smart meter voltage measurement. Each column is a smart meter. This is the noiselss smart meter data to calculate ground truth accuracy. If noiseless data is not available, then this table can be the same as "smart_meter_volt.csv".


2. Run the main code "main_NN_sincos_VI_noisy.py" performs the following works:
(1) Split the input and output data of the model into training, validation, and testing datasets. The input and outpu data are provided by the user.
(2) Train a neural network model based on the training data. The training is stopped by early stopping based on the validation performance on the validation dataset. The performance of MAPE is then evaluated on the testing dataset.
(3) For each available DPMU loaction (combining both voltage location and current location), a model is trained and the MAPE is collected.
(4) The code then save a summary csv file. In the file, each row is a DPMU location, and it shows the MAPE of each smart meter's voltage estimation, and the overall MAPE of each row.

3. User need to set/define the following variables or paths in the main code "main_NN_sincos_VI_noisy.py":
(1) table_path: This is the path of the folder that contains the needed data files.
(2) temp_file_path: This is the path and file name of the saved temporary best model file during training.
(3) save_MAPE_csv_table_path: The path and file name of the saved summary csv file.
(4) epoch_num: Maximum number of epochs in the model training.
(5) batch_size: Number of samples in a mini-batch.
(6) patience: Early stopping patience, in terms of epochs.

