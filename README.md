# Chemical and olfactory models for pre-training VOC sensors

This is the code for work "Chemical and olfactory models for pre-training VOC sensors" done by Hitesh Sapkota during Internship 2022 under the supervision of Dr. Wesley Hong. 
Because of the Confidential Issue, the detailed description of this work is not provided here. Please refer to the following documents if you are interested in this work 
and talk to the Wesley to get the document permission.

### Initial Project Plan Document: https://quip-amazon.com/YAkzAQzArNaq/Project-Plan
### Midpoint Document: https://quip-amazon.com/nlYPAT8TJkE2/Midterm-Model-Evaluation

## Dependencies:
Python3 Installation
GPU Recommended
Execute following Commands to install all dependent python packages
python -m pip install -r requirement.txt

## Training:
  The current version supports: (a) baseline training, (b) structure finetuning training, (c) attention (with gated or basic) based training, and (d) attention+finetuning training. To train the model go to src/scripts/experiments then run either train.py or train_s3.py. Both scripts are identical except train_s3 performs data read/write, model read/write operations from/to s3 bucket specified as an argument whereas, train.py performs all operations relative to the current train.py script path. Throughout this readme file, when we say root_dir it means the directory same as that of the train.py in the case we are dealing with train.py and root_dir means bucket_name/bucket_folder in s3 bucket whenever we are dealing with train_s3.py. All data, models, and outputs will be read and write relative to this root_dir.
  
  Common Arguments for both scripts are as follow
  
  ### Common Arguments:
  **Data Related Arguments**
  - data_dir: Unprocessed input data directory to be used (Usually it is '../../sniff_ml/data_utils/datasets/data/bushdid' for bushdid data)
  - output_dir: Output to be stored (mainly inference result, training/validation loss) (Usually it is ./outputs/structure_encoder-odor_encoder)
  - n_fold: Number of Folds used for training/testing split. Number of training/testing set replication
  - valid_size: Percentage of Training set used for hyperparameter tuning such as early stopping
  - rep_no: replication (fold) number used during training from a given folds. Value stays between [1..., n_fold} 
  - num_workers: Number of workers used for the training process.
  - batch_size: Number of data samples in each step
  - rand_state: Random seed used to make the replication. Usually value between 1-5 is set.
  
  **Network Architecture Related Arguments**
  - odor_encoder: Type of Odor Encoder used, Choices ='OdorEncoder1', 'OdorEncoder2'
  - odor_embed_size: Oder Encoder Embedding Size.
  - structure_encoder: Type of the attention performed to aggregate the molecule representation to get the mixture moelcule representation. Choices =           'unweighted', 'weighted'
  - structure_encoder_type: Type of structure encoder used. Choices = 'mol2vec', 'molembtrans', 'molbert'
  - structure_embed_size: size of the structure encoder (look at the pretrained network to set the size)
  - attention_type: Type of attention used to combine molecules to get the mixture-molecule representation. Choices = 'basic', 'gated'
  - fintune: Whether we want the structure encoder to be finetuned or fixed. Default True
  - odor_finetune: Whether we want the odor encoder to be trained with different learning rate. Default False
  - gpu_use: Whether we want the gpu to train the model. Choices = 'cuda', 'cpu'
  
  **Optimization Related Arguments**
  - loss_margin: Loss margin for the contrastive loss. Default 2
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
  - Checks whether processed data is already there for a given rep_no and soft_label setting. If exists, it simply reads data through data folder from root_dir.
  - If data is not present, it calls bushdid.py scripts which reads raw data through the folder ../../sniff_ml/data_utils/datasets/data/bushdid.         bushdid.py script processes the data and stores into the data folder present in root_dir. If data folder is not already present in the root_dir, it creates it and stores the data. Specifically, the data: (a) fit_samples_rep_no.pickle, (b) valid_samples_rep_no.pickle, (c) test_samples_rep_no.pickle, where rep_no is the fold (replication) number, will be stored. In case of soft label setting the data: (a) fit_samples_rep_no_softlabel.pickle, (b) valid_samples_rep_no_softlabel.pickle, (c) test_samples_rep_no_softlabel.pickle will be stored. Also, mol_atom_map.pickle will be store to have mapping between moleucles with associated atoms. Do each file will be under root_dir/data.
  - Next, the scripts creates train and validation loader and performs training by calling the SiameseNet. Early stopping is performed during training by tracking the validation loss. The model with lowest validation loss is used as the best model and last epoch model is treated as last model. 
  - After the completion of training, the inference is performed in the training, validation, and testing dataset set and the corresponding output is generated.
  - Inference output (training, validation, test) along with the training, and validation losses will be stored in the output_dir inside root directory Specifically, losses are stored in root_dir/output_dir/losses/training_identifier with names training_loss.pickle, val_loss.picke. Inference outputs are stored inside  root_dir/output_dir/results/ with names train_prediction_training_identifier.pickle, valid_prediction_training_identifier.pickle,                          test_prediction_training_identifier.pickle. The training_identifier holds the argument confifuration we pass as an input. output_dir by default will be ./outputs/structure_encoder-odor_encoder with structure_encoder and odor_encoder as an input arguments.
  - The best model and last models will be stored in root_dir/models/training_identifier+_best.pth and root_dir/models/training_identifier+_last.pth respectively.

## Performance Generation:
To generate the performance in terms of AUC in case of hard label sampling and MAE in case of soft label setting, we can run the scripts analysis_all_models.py. It has following input arguments
### Input Arguments:
- bucket_name: name of the s3 bucket if read the outputs from the s3 bucket. Default piml-sniff
- bucket_folder: Name of the folder inside s3 bucket that can be used as root working directory to read/write data/models
- data2s3: Whether to read outputs (prediction scores) from the s3 bucket. Default True
### Code Specific Arugments:
Other than the input arguments, you can directly change some of the arguments in the code. For example, for the soft_label make it to True. Most of arguments are same as that of the one used during the training process.

### Running:
Run the code analysis_all_models.ipynb that calls analysis_all_models.sh. In analysis_all_models.sh the input arguments are passed to the code analysis_all_models.py. Finally, analysis_all_models performs the following:
1. To generate the performance it reads train_prediction_identifier.pickle, valid_prediction_identifier.pickle, test_prediction_identifier.pickle from root_dir/output_dir/results.
2. For each fold, get  performance for all settings like baseline, attention, finetune, attention+finetune. It generates the mean and sd performance done over all folds, all rand_state for all settings and store the result  dataset_type+.csv in root_dir/outputs folder. In case of soft label setting it store the result dataset_type+_softlabel.csv inside root_dir/outputs/. Here, dataset_type can be 'train', 'valid', or 'test'. The script will store the result for all dataset_type. 

 


