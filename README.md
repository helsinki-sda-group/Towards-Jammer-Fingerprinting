#### Towards jammer fingerprinting

This code is used for test setups in the paper:
O. Savolainen, T. Malmivirta, Z. Yan, A. Morrison and L. Ruotsalainen, "Towards Jammer Fingerprinting: The Effect of the Environment and the Receiver to a Jammer Classification," 2024 International Conference on Localization and GNSS (ICL-GNSS), Antwerp, Belgium, 2024, pp. 1-7, doi: 10.1109/ICL-GNSS60721.2024.10578380.


##### Data
- This code expects that the data is in a binary file (`.dat`) in a in-phase and quadrature (IQ) format. These components should alternate as follows IQIQIQ...
- The code expects that the file name is something like this `j11a-20240109-17-02-35-50e6-50e6.dat` having the recording bandwidth just before extension (`.dat`)
- the folder containing the binary files contains the name of the jammer and whether it has GNSS signal included or not `Jammer11aNoGNSS` or `Jammer11a`
- give the parent path for the folder containging the different experiments, e.g., `/my_path/fingerprinting_data/`
- give the files to read as input as follows `usrp1/*/*.dat usrp2/*/*.dat usrp3/*/*.dat usrp4/*/*.dat usrp5/*/*.dat usrp6/*/*.dat` which allows to read the files for all the different jammer devices. The code expects 6 different paths, change the `runs.py` if the different number of is needed.
