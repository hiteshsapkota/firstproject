# Chemical and olfactory models for pre-training VOC sensors

This is the code for work "Chemical and olfactory models for pre-training VOC sensors" done by Hitesh Sapkota during Internship 2022 under the supervision of Wesley Hong (Manager), saravanan sathananda manidas (Mentor), and Hongtao Yu (Onboarding Buddy). 
Because of the Confidential Issue, the detailed description of this work is not provided here. Please refer to the following documents if you are interested in this work and talk to Wesley to get the document permission.

### Initial Project Plan Document: https://quip-amazon.com/YAkzAQzArNaq/Project-Plan
### Midpoint Document: https://quip-amazon.com/nlYPAT8TJkE2/Midterm-Model-Evaluation
### Final Project Documet: https://quip-amazon.com/tpJjArJYxoA7/Toward-Protocols-to-Improve-Molecule-Embedding-on-Odor-Discrimination-Task

## Dependencies:
Python3 Installation

GPU Recommended

Execute following Commands to install all dependent python packages

python -m pip install -r requirements.txt

## Training:
Training involves: (a) Single Task Training (with Bushdid Dataset), and (b) Multitask Learning (with Multiple Datasets).


**The Single Task learning**:
It supports (i) baseline training, (ii) structure finetuning training, (iii) attention (with gated or basic) based training, and (iv) attention+finetuning training. We can perform training either using sagemaker studio or sagemaker api.
 
 ### Using Sagemaker Studio:
 To train the model go to src/experiments/training then run train.py using either of the scripts baseline.ipynb, finetune.ipynb, attention.ipynb, finetune_attention.ipynb. Depending on the input argument called use_bucket, the code performs all data/model/output read write operation from either local directory or through s3 bucket. If use_bucket is true then the code performs data read/write, model read/write operations from/to s3 bucket specified as an argument whereas, if false performs all operations relative to the current train.py script path. Throughout this readme file, when we say root_dir it means the directory same as that of the train.py in the case we are dealing with use_bucket = False and root_dir means bucket_name/bucket_folder in s3 bucket whenever we are dealing with use_bucket=True. All data, models, and outputs will be read and write relative to this root_dir. 

### Using Sagemaker API:
A separate train_sagemaker_api.ipynb the peforms training  for baseline, structure encoder finetuning, attention, and finetuning+attention. Same training script is used as that of training using sagemaker studio except we set input argument use_sagemaker_studio = True in this case.
    
    
**Multitask Learning:** 
Our Multitask setting supports: (i) baseline training, and (ii) attention+finetuning training. We can perform training either using sagemaker studio or sagemaker api.
### Using Sagemaker Studio:
    To train the model go to src/experiments/training then run train_multitask.py using baseline_multitask.ipynb or finetune_attention_multitask.ipynb. This will similar to the single task Learning except it takes multitple datasets and performs multitask learning training.
    
### Using Sagemaker API:
To train the model go to src/experiments/training and then run train_multitask.py using script train_sagemaker_api_multitask.ipynb. It is identical to using Sagemaker studio except we need to set argument use_sagemaker_api = True
  
Common Arguments for Training Scripts
  
  ### Common Arguments:
  **Data Related Arguments**
  - data_dir: Unprocessed input data directory to be used (Usually it is '../../sniff_ml/data_utils/datasets/data/bushdid' for bushdid data)
  - output_dir: Output to be stored (mainly inference result, training/validation loss) (Usually it is ./outputs/structure_encoder-odor_encoder)
  - n_fold: Number of Folds used for training/testing split. Number of training/testing set replication
  - valid_size: Percentage of Training set used for hyperparameter tuning such as early stopping
  - rep_no: replication (fold) number used during training from a given folds. Value stays between 1,..., n_fold 
  - num_workers: Number of workers used for the training process.
  - batch_size: Number of data samples in each step
  - rand_state: Random seed used to make the replication. Usually value between 1-5 is set.
  - use_bucket: Whether to use the s3 bucket for all data related operations. Default True.
  - use_sagemaker_api: Whether to use the Sagemaker API for model training and prediction. Default False.
  - bucket_name: Name of S3 bucket we will use to read and write files/models/data. Default: piml-sniff
  - bucket_folder: Root folder (directory) inside S3 which will be used to store all results, models, data. 
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
 

  ### Multitask Learning Training Related Arguments
  - des_loss_coeff: Descriptor Task Loss Weight. Default 0. Input agrument will be divided by 100.
  - od_loss_COcoeff: Odor Prediction Loss Weight. Default 0. Input argument will be divided by 100.
  

  
  ### Running:
  Before running, if you want to run for the structure encoder 'molembtrans', then please download the pretrained network from the link wget https://github.com/mpcrlab/MolecularTransformerEmbeddings/releases/download/checkpoints/pretrained.ckpt. 
  Once download is completed, upload it to the foloder 'sniff_ml/ml/structure_encoder/pretrained' with model name as 'molembtrans_model_512dim.ckpt'.
  If you want to run the structure encoder 'bert', then please download the pretrained network from the link wget https://ndownloader.figshare.com/files/25611290.
  Once download is completed, upload it to the folder 'sniff_ml/ml/structure_encoder/pretrained' with model name as 'bert_model_768.ckpt' and .yaml file to hparams.yaml.
  
 In case of single task learning, we have provided separate sagemaker scripts to run: (a) Baseline (baseline.ipynb), (b) Structure Encoder Finetuning (finetune.ipynb), (c) Attention (attention.ipynb), (d) Structure Encoder Finetuning+Attention (finetune_attention.ipynb). In case of Multitask Learning two scripts: (a) Baseline (baseline_multitask.ipynb), and (b) Structure Encoder Finetuning+Attention (finetune_attention_multitask.ipynb) are provided. Each script performs following. 
  1. Calls respective .sh script. For instance, baseline.ipynb calls baseline.sh command.
  2. .sh script calls train.py  along with the arguments required for the specific setting. Feel, free to e 
     explore the used arguments. It is noted that you do not need to add any additional argument for the specific setting. However, you can change the          value of argument. Each .sh has rep_no and rand_state variabels, rep_no indicates the fold number (data split) number and rand_state indicates the          random seed used to initialize random variables such as network intialization. rep_no should be between 1-5. 
  3. train.py  performs following:
  - Checks whether processed data is already there for a given rep_no and soft_label setting. If exists, it simply reads data through data folder from root_dir.
  - If data is not present, it calls associated data (bushdid for single task setting and multiple for multitask setting), processes the data and stores into the data folder present in root_dir. If data folder is not already present in the root_dir, it creates it and stores the data. Specifically, concerning single task setting the data: (a) fit_samples_rep_no.pickle, (b) valid_samples_rep_no.pickle, (c) test_samples_rep_no.pickle, where rep_no is the fold (replication) number, will be stored. In case of soft label setting the data: (a) fit_samples_rep_no_softlabel.pickle, (b) valid_samples_rep_no_softlabel.pickle, (c) test_samples_rep_no_softlabel.pickle will be stored. Also, mol_atom_map.pickle will be store to have mapping between moleucles with associated atoms. Do each file will be under root_dir/data. In case of multitask setting, filename will be same except multitask term such as fit_samples_rep_no_multitask.ipynb. 
  - Next, the scripts creates train and validation loader and performs training by calling the SiameseNet. Early stopping is performed during training by tracking the validation loss. The model with lowest validation loss is used as the best model and last epoch model is treated as last model. 
  - After the completion of training, the inference is performed in the training, validation, and testing dataset set and the corresponding output is generated.
  - Inference output (training, validation, test) along with the training, and validation losses will be stored in the output_dir inside root directory Specifically, losses are stored in root_dir/output_dir/losses/training_identifier with names training_loss.pickle (training_loss_multitask.pickle), val_loss.picke. Inference outputs are stored inside  root_dir/output_dir/results/ with names train_prediction_training_identifier.pickle (training_prediction_training_identifier_multitask.pickle), valid_prediction_training_identifier.pickle,                          test_prediction_training_identifier.pickle. The training_identifier holds the argument confifuration we pass as an input. output_dir by default will be ./outputs/structure_encoder-odor_encoder with structure_encoder and odor_encoder as an input arguments.
  - The best model and last models will be stored in root_dir/models/training_identifier+_best.pth and root_dir/models/training_identifier+_last.pth respectively.
  
**Note: In case of training using sagemaker_api, all data will be stored to the S3 bucket obtained using 'bucket = sagemaker_session.default_bucket()' under the folder name 'sagemaker'. All other operations are exactly same.**

## Performance Generation:
To generate the performance in terms of  Mean Absolute Error (MAE) in case of soft label setting, we can run the multiple scripts located under experiments/evaluation to generate different tables. 


### analysis_all_models.py:
Considers the Single Task Setting, and generates the average MAE score (for soft_label) averaged over 5 fold data and 5 replications. Running this script will generate result for the training, validation, and testing data and stores in directory root_dir/outputs/ with name train_soft_label_single_task.csv, valid_soft_label_single_task.csv, and test_soft_label_single_task.csv. If soft_label is False stores, train_single_task.csv, valid_single_task.csv, and test_single_task.csv. This will generate Table 1 in the Final Project Document.

### inference_multitask_bushdid.py:
This will generate the output prediction result using multitask learning learned model related to the bushdid dataset. 

### analysis_all_models_multitask.py: 
Considers the Multitask Setting, and generate the average MAE score (for the data associated with snitz, bushdid, and ravia) averaged over 5 fold data and 1 replication. Running this script will generate result for the training, validation, and testing data and stores in the directory oot_dir/outputs. If we set use_buhdid_only = True, then generates the result considering only bushdid dataset used in the multitask learning. 
. 
**To generate the Table 3 (last column) of final document, run the analysis_all_models_multitask.ipynb with  input argument use_bushdid_only = True and des_loss_coeff = 0, od_des_coeff = 0.** 

**To generate the Table 4 of final document, run the analysis_all_models_multitask.ipynb with  input argument use_bushdid_only = False and des_loss_coeff = 10, od_des_coeff = 10.**

**To generate the Table 5 (last column) of final document,  run the analysis_all_models_multitask.ipynb with  input argument use_bushdid_only = True and des_loss_coeff = 10, od_des_coeff = 10.**

### analysis_structure_encoders.py:
Considers the 'mol2vec', 'molembtrans', and 'molbert' structure encoders and generates the MAE score (averaged over 5 folds). This will generate Table 2 
in the Final Project Document.
