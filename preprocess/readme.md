# Predict free-text reports from X-ray image
[RATCHET](https://github.com/farrell236/RATCHET) works as a baseline to be compared with our method. In this git repo you 
could find the pretrained model and predict free-text reports from X-ray images.

# Preprocess free-text reports
Once you have the free-text reports, you need the pretrained RadGraph model in [RadGraph Benchmark](https://physionet.org/content/radgraph/1.0.0/) to **convert the free-text reports to structured graph json files**.

# From structured graph json file to dataset 
It is not enough to have just the RadGraph-format json files, you need following script to build the final dataset.

[OBS_ANAT_list.json](OBS_ANAT_list.json) lists all selected observations and anatomies.
[gen_dataset.py](gen_dataset.py) generates the dataset used in our model, while the input is the radgraph-format json file.
It converts all entities names into non-sensitive numbers to input the model.

# Pre-save the X-ray images locally
To save the loading resources, we pre-transform the images and save them as tensors locally. You could find more details in the 
[script file](preprocess_img.py)
