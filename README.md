# MSNet-4mC

Author: Liu Chunting
Affiliation: Department of Intelligence Science and Technology, Graduate School of Informatics, Kyoto University
E-mail: liuchunting@kuicr.kyoto-u.ac.jp

## Details
* Users can run the MSNet_4mC.py to identify DNA 4mC sites.  
* The folders of “Li_2020_dataset” and “Lin_2017 dataset” contain the datasets and the files for experiments.  
* The Lin_2017 dataset and Li_2020 dataset can be also accessed at http://DeepTorrent.erc.monash.edu/  
* The trained model weights for the test are provided in the folder ‘Models’.  

## Dependency
* Python 3.8.8 and Pytorch 1.11.0 or later versions  

## Installation Guide
* Download from GitHub at https://github.com/LIU-CT/MSNet-4mC  
* cd MSNet-4mC  
* pip install -r requirements.txt  

## Usage
### For evaluation:
* Run the default dataset  
```
python MSNet_4mC.py --Dataset Lin_2017_Dataset --Species <Species>
``` 
```
python MSNet_4mC.py --Dataset Li_2020_Dataset --Species <Species> 
```
* Make the prediction for customized data  
```
python MSNet_4mC.py --Dataset User_Dataset --Species <Species> --Fasta_file <Fasta_file>  
```
* For evaluation on the default dataset, users can also directly run “test.py” with corrected paths to the datasets and models.   
### For training
* Train the default dataset  
  Run “training_basemodel.py” and “training_scratch_and_finetuning.py”.  
* Train the customized data  
  Firstly, use “class_weight.py” to calculate class weights for different species.  
  Then, use “training_basemodel.py” to train a base model on the merged training dataset.  
  Finally, use “training_scratch_and_finetuning.py” to retrain the species-specific model on each species training dataset.


