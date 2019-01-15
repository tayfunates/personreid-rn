# personreid-rn
Market1501 Person Re-Identification Task Using Relational Networks

./src folder contains a couple of source files for the implementation.

./src/ReidRN.py contains a relation network which calculates a similarity of two image/feature pair input 
./src/ReidRNSelf.py contains a relation network which calculate a feature for a single image/feature input
./src/dataset_loader.py contains loader classes for images or pre-extracted features.  
./src/eval_metrics.py contains evaluation function for Market1501.  
./src/extract_features.py can be used to extract features from images using Resnet.  
./src/market1501_rn.py contains the dataset class.  
./src/samplers.py contains batch samplers. Pairwise sampling is used to learn the relations between different identities.  
./src/train_rn.py is the script for training and testing, using image pairs to extract relations.
./src/train_rn_self.py is the script for training and testing, using a single image to extract the relations
./src/utils.py contains some utility functions.  
./src/losses.py contains different types of loss function classes

This work is inspired by the work of [Santoro et al.](https://arxiv.org/abs/1706.01427)
