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
  
  ### Common Arguments:
  **Data Related Arguments**
  - data_dir: Unprocessed input data directory to be used (Usually it is '../../sniff_ml/data_utils/datasets/data/bushdid' for bushdid data)
  - output_dir: Output to be stored (mainly inference result, training/validation loss) (Usually it is ./outputs/structure_encoder-odor_encoder)
  - n_fold: Number of Folds used for the testing
  - valid_size: Percentage of Training set used for hyperparameter tuning such as early stopping
  - rep_no: replication (fold) number used during training 
  - num_workers: Number of workers used for the training process.
  - batch_size: Number of data samples in each step
  - rand_state: Random seed used to make the replication
  
  **Network Architecture Related Arguments**
  - odor_encoder: Type of Odor Encoder used, Choices ='OdorEncoder1', 'OdorEncoder2'
  - odor_embed_size: Size of the Odor encoder.
  - structure_encoder: Type of the attention performed to aggregate the molecule representation to get the mixture moelcule representation. Choices =           'unweighted', 'weighted'
  - structure_encoder_type: Type of structure encoder used. Choices = 'mol2vec', 'molembtrans', 'molbert'
  - structure_embed_size: size of the structure encoder (look at the pretrained network to set the size)
  - attention_type: Type of attention used to combine molecules to get the mixture-molecule representation. Choices = 'basic', 'gated'
  - fintune: Whether we want the structure encoder to be finetuned or fixed. Default True
  - odor_finetune: Whether we want the odor encoder to be trained with different learning rate. Default False
  - gpu_use: Whether we want the gpu to train the model. Choices = 'cuda', 'cpu'
  
  **Optimization Related Arguments**
  - loss_margin: Loss margin for the contrastive loss
  - learning_rate: Network optimization learning rate except for the structure encoder. Input learning rate will be divided by 10**-05
  - structure_learning_rate: Structure Encoder learning rate. Input learning rate will be divided by 10**-05
  - soft_label: Whether Soft Label setting is considered for the training. Default False
  - soft_loss: Type of loss used for the soft label training. Choices = 'mae'
  - max_epochs: Total Maximum Epoch considered during training.

  ### train_s3 specific arguments:
  - bucket_name: Name of S3 bucket we will use to read and write files/models/data. Default: piml-sniff
  - bucket_folder: Root folder (directory) inside S3 which will be used to store all results, models, data. 

  
  ### Running:
  We have provided separate sagemaker scripts to run: (a) Baseline (baseline.ipynb), (b) Structure Encoder Finetuning (finetune.ipynb), (c) Attention (attention.ipynb), (d) Structure Encoder Finetuning+Attention (finetune_attention.ipynb). Each of the script will do following:
  1. Calls respective .sh script. For instance, baseline.ipynb calls baseline.sh command.
  2. .sh script calls train.py or train_s3.py (you can change it in .sh script) along with the arguments required for the specific setting. Feel, free to e 
     explore the used arguments. It is noted that you do not need to add any additional argument for the specific setting. However, you can change the          value of argument. Each .sh has rep_no and rand_state variabels, rep_no indicates the fold number (data split) number and rand_state indicates the          random seed used to initialize random variables such as network intialization. rep_no should be between 1-5. 
  3. train.py or train_s3.py performs following:
  - Checks whether processed data is already there for a given rep_no and soft_label setting. If there, it simply reads data through data folder. In case       of train_s3.py, data folder is read through S3 bucket_Name/ bucket_folder and in case of train.py, data folder is present in the directory same as that     of train.py.
  - If data is not present, then it calls bushdid.py scripts and it reads raw data through the folder ../../sniff_ml/data_utils/datasets/data/bushdid.         bushdid.py scrip processes the data and stores into the data folder either in the directory same as train.py or in bucket_name/bucket_folder (in case       of train_s3.py).
  - Next, the scripts creates train and validation loader and performs training by calling the SiameseNet. Early stopping is performed during training by tracking the validation loss. The model with lowest validation loss is used as the best model and last epoch model is treated as last model. 
  - After the completion of training, the inference is performed in the training, validation, and testing set and the corresponding output is generated.
  - Inference output (training, validation, test) along with the training, and validation loss are stored in the output_dir with the root directory as         directory that of train.py in case of train.py and bucket_name/bucket_folder as root directory in case of train_s3.py. Specifically, losses are stored     in root_dir/output_dir/losses/training_identifier with names training_loss.pickle, val_loss.picke. Inference outputs are stored inside                     root_dir/output_dir/results/ with names train_prediction_training_identifier+.pickle, valid_prediction_training_identifier+.pickle,                          test_prediction_training_identifier+.pickle. The training_identifier holds the argument confifuration we pass as an input
  - The best model and last models are stored in root_dir/models/training_identifier+_best.pth and root_dir/models/training_identifier+_last.pth with root directory same as that of train.py in train.py and bucket_name/bucket_folder in case of train_s3.py
  -
  


