# State-estimation-by-DPMU
State estimation by DPMU

This project uses a neural network model for state estimation of the distribution system. The input and output are as follows:
Input of the model: hour and hour data in cyclical encoding (cos,sin); weekday/weekend, holiday; voltage and current measurement of a DPMU.
Output of the model: the voltage magnitude of the feeder's smart meters.

Both the DPMU and smart meter data contains noise. However, when evaluating the testing performance, we should use noiseless smart meter data for reference.

