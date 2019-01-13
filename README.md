# personreid-rn
Market1501 Person Re-Identification Task Using Relational Networks

./src folder contains a couple of source files for the implementation.

./src/ReidRN.py contains relation network of Santoro et al.  
./src/dataset_loader.py contains loader classes for images or pre-extracted features.  
./src/eval_metrics.py contains evaluation function for Market1501.  
./src/extract_features.py can be used to extract features from images using Resnet.  
./src/market1501_rn.py contains the dataset class.  
./src/samplers.py contains batch samplers. Pairwise sampling is used to learn the relations between different identities.  
./src/train_rn.py is main script for training and testing.  
./src/utils.py contains some utility functions.  
