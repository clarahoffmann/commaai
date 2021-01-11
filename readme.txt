Marginally calibrated posterior densities for end-to-end learning
- Code Overview

Master thesis by Clara Hoffmann
last changed: 15.01.2021

The code serves to replicate the results in the thesis. Note that due 
to storage limitations, the code starts after the comma.ai 2k19 video data
has been transformed to a frame-by-frame-basis. If users want to replicate this rather trivial step
they can download the entire comma.ai 2k19 data set from 

https://academictorrents.com/details/65a2fbc964078aff62076ff4e103f18b951c5ddb

which is ca. 100GB. Then extract all video files into a single folder.
The pruned data set was created by manually labelling one fifth of the entire 
data set using a web application. The resulting training indices can be found 
in the file 'data/commaai/training_files_filtered/indices/train_indices.csv'.
This took approximately 40 hours.

The code is structured as follows:

(0. 'sort_into_bags_optional' (optional):
    The comma.ai 2k19 video data is reduced to frames. Each frame is sorted
    into a folder based on the associated steering angle. For obtaining appropriate
    pruning and oversampling, observations are sampled from these folders
    to create the training shards (see step 3). To run this the full comma.ai data set
    has to be downloaded as described above. This step is time and storage intensive.)

1. '01_density':
    Estimates the density for the pruned and unpruned data set.

(2. '02_write_shards_optional' (optional):
    Creates the training and validation shards for the pruned and unpruned data. 
    Training observations are sampled at random from the folders in step 1.
    Shards are tfrecords files, that save the images and associated steering angles 
    in binary format to save storage and speed up reading in the data while training the
    end-to-end learners. This step is very time intensive, so it is recommended to use
    the already created shards. )

3. '03_models':
    Trains the end-to-end learners. Extracts the basis functions for the precise and
    imprecise learners.

4. '04a_MCMC':
    HMC estimation for the CPL/Ridge, CPL/Horseshoe and CIL/Ridge, CIL/Horseshoe.
   
   '04b_VA':
    VAFC estimation for the CPL/Ridge, CPL/Horseshoe and CIL/Ridge, CIL/Horseshoe.
    
5. '05_predictions':
    Creates calibration plots, explainability plots, accuracy plots for HMC vs. VAFC, qqplots,
    validation performance.






