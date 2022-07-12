# Chemical and olfactory models for pre-training VOC sensors

This is the code for "Chemical and olfactory models for pre-training VOC sensors" done by Hitesh Sapkota during Internship 2022 under Wesley Hong's supervision.
Because of the Confidential Issue, the detailed description of this work is not provided. Please refer to the following documents if you are interested in this work 
and talk to the Wesley to get the document permission.
### Initial Project Plan Document: https://quip-amazon.com/YAkzAQzArNaq/Project-Plan
### Midpoint Document: https://quip-amazon.com/nlYPAT8TJkE2/Midterm-Model-Evaluation

## Dependencies:
Python3 Installation
GPU Recommended
Execute following Commands to install all dependent files
python -m pip install -r requirement.txt

## Training:
  The current version supports: (a) baseline training, (b) structure finetuning training, (c) attention (with gated or basic) based training, (d) attention+finetuning training. To train the model go to src/scripts/experiments then run either train.py or train_s3.py. Both scripts are identical and only the difference is that train_s3 performs data read/write, model read/write operations from/to s3 bucket specified as an argument whereas, train.py performs all operation relative to the current train.py script path. Common Arguments for both scripts are as follow
  
  ### Arguments:
  {\bf Data Related Arguments}
  train.py or train_s3.py 


