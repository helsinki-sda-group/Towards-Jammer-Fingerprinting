## Towards jammer fingerprinting

This code is developed for the test setups used in the paper:
O. Savolainen, T. Malmivirta, Z. Yan, A. Morrison and L. Ruotsalainen, "Towards Jammer Fingerprinting: The Effect of the Environment and the Receiver to a Jammer Classification," 2024 International Conference on Localization and GNSS (ICL-GNSS), Antwerp, Belgium, 2024, pp. 1-7, doi: 10.1109/ICL-GNSS60721.2024.10578380.

Includes two Long Short-term Memory network -based architectures for a classification task: bayesian and 'traditional' one.

## Paper abstract
Global Navigation Satellite Systems (GNSS) are utilized in many fields in everyday life. However, the number of radio-frequency interference (RFI) events, especially intentional ones, has been growing in recent years on the busy roads [1]. Such intentional interference is mainly jamming, meaning the transmission of radio signals at the GNSS frequency bands burying the real signal. The use of jamming devices is illegal in most countries, however, these jammers are easily bought from the internet and simple to use. Machine learning algorithms for radio-frequency signal classi-fication and fingerprinting exist. Each jammer has its own signal characteristics due to, for example, the non-linearities in the components arising from the manufacturing process [2]. However, the recording environment is not exactly the same at different times even at the same location. In order to reliably pair the signals with the transmitting devices, modeling the possible effects of the environment is important. To analyze these effects, we apply a baseline LSTM architecture to classify different jammers based on their transmitted signals. Also, Bayesian LSTM-based approach is introduced to model uncertainty in the predictions, that is, model's confidence. We train and test both networks with datasets acquired with different radio front-end devices in an anechoic chamber, first via transmitting only a jamming signal, and then together with GNSS signal repeated from an outdoor antenna. The understanding of environmental effects is the first step for developing robust fingerprinting methods. The main contribution of this paper is the analysis of the results that is further applicable for the next research step, namely the development of domain adaption algorithms to generalize the method over the effects.

## Data
- Data is expected to be in a binary file (`.dat`) including in-phase and quadrature (IQ) components. These two components should alternate as follows IQIQIQ...
- File name is expected to be similar to `j11a-20240109-17-02-35-50e6-50e6.dat` having the recording bandwidth just before the file extension (`.dat`). The bandwidth should be presented with scientific notation, for example **0.1e6** or **3e6**.
- Folder containing the binary files should contain the name of the jammer and whether it has GNSS signal included or not `Jammer11aNoGNSS` or `Jammer11a`
    - In general, the meaning of **noGNSS** is that the signal is recorded through cable. However, some environments require, that all data is recorded through cable, and on the other hand, jammers with integrated antenna without possibility       to attach a cable, are then recorded by using antenna.
    - All data used in our experiments were collected in the EMC chamber with a permission, and limited transmitting power.
    - The data collected by us is not public available.

## Running the program (`runs.py`)
Run `runs.py` program either from console or if jupyter notebook is installed, one can run the code also in the interactive window by clicking _run below_ in the first line. Follow the instructions visble in the console or interactive window:
- First asked input: give the path for the parent folder containging the jammer specific data folders, e.g., `my_path/fingerprinting_data/`
- Second asked input: To read all files for different devices at once, you should give all 6 (ecpected, but changeable in `runs.py`) paths, e.g., `usrp1/*/*.dat usrp2/*/*.dat usrp3/*/*.dat usrp4/*/*.dat usrp5/*/*.dat usrp6/*/*.dat`.
In need to change anything in file names etc, check `runs.py`. To fix the **class names** to correspond the correct ones, go to `enums.py`.

## Requirements
General machine learning, signal and data processing libraries
- numpy
- pandas
- scikit-Learn
- PyTorch
- seaborn
- scipy

For plotting
- seaborn
- matplotlib

Bayesian network
- blitz

## License
This code is licened with a MIT license.
