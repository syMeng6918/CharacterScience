This repository contains code used for the Newcomers' Assignment in the Intelligent Media Processing Lab.  
The task is to train a binary classifier to detect whether a font matches a specific impression tag (e.g., "decorative", "condense").

## Environment(Temporary)
- Please use the Dockerfile to build the environment.
- docker build -t font-trainer . 
- sudo docker run --shm-size=8g --gpus all -it -v $(pwd):/workspace font-trainer
 
## Main Files

- `train_multimgv4.py`: Main training script. Supports command-line arguments such as model type, loss function, target tag, and number of letters.
- `preprocess_data_reference.py`: Revised version of the original preprocessing script, modified to be compatible with updated libraries.
- `dataset.py`: Dataset class for **single-character** input training.
- `dataset_repeat.py`: Dataset class for **multi-character** input. Supports dynamic repetition (i.e., synthesizing multiple samples per font per epoch).

## Folder

- `dataset/`: Contains:
  - Preprocessed training images  
  - CSV files for font-tag mappings  
  - A frequency statistics file used to decide which tags are worth testing


## How to start training
- python3 train_multimgv4.py --tag decorative --epochs 60 --lr 0.002 --warmup_start_lr 0.0005 --model efficientnet --batch_size 32
- You can change the tag to others like condense, animate and any other paramaters, via the --tag argument.
- 💡 If the dataset is highly imbalanced, you can adjust the gamma_neg parameter in the ASL loss function (e.g., set gamma_neg=1.5) to improve robustness against dominant negative samples.



