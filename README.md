# recursive_chemprot
This is the open source repository of my Recursive Neural Network model for the BioCreative 6 track5 - chemprot task.  
The paper is located at: https://academic.oup.com/database/article/doi/10.1093/database/bay060/5042822  
Our experiments from the paper are divided into 3 approaches.  
The first approach use the tree-LSTM model and this approach was used in the CHEMPROT challenge.  
After the challenge, we experimented two enhacements. The second approach applied additional preprocessing steps to our tree-LSTM model, and the third approach tested the performance of the another Recursive Neural Network model, SPINN.  

We distribute the newly preprocessed data only (2nd and 3rd approaches).  
We provide the two enhancement approaches with different directory.  
## Requirements:  
tree-LSTM  
&ensp;Tensorflow ver 1.0  (we do not test the other versions, we found that tensorflow fold is known to be working best at tf version 1.0.0.)  
&ensp;Tensorflow Fold https://github.com/tensorflow/fold  
  
SPINN  
&ensp;Tensorflow ver 1.8  (eager mode is required) 
  
Common  
&ensp;python 3.4 or later  
&ensp;&ensp;gensim https://radimrehurek.com/gensim/install.html  
&ensp;&ensp;sklearn http://scikit-learn.org/stable/install.html  
&ensp;&ensp;numpy  
&ensp;&ensp;nltk  
  
## Data:  
&ensp;All the data for the model is inside the data folder of the the each approaches. Due to the file sizes, we zipped the files.  
&ensp;Extract the data zipped files inside the data folder. All data is not the original data.(pre-processed data)  
&ensp;Word embedding is not included. visit : http://evexdb.org/pmresources/vec-space-models/  
&ensp;Original Challenge data is in the biocreative site : http://www.biocreative.org/resources/corpora/chemprot-corpus-biocreative-vi/ (login required)  
&ensp;*update: biocreative site has been changed to ==> https://biocreative.bioinformatics.udel.edu  
&ensp;*update: original biocreative site is opened again!  

## Training and Testing:  
tree-LSTM  
&ensp;First, run the "saveWEUsedinDataOnly.py" code to reduce the size of the word embedding. Our code assumes that the original word embedding file is located in the folder "bioWordEmbedding" and the shortened word embedding file is located inside the "shorten_bc6" folder.  
&ensp;Second, run the "BC6track5Recursive.py" code for the treeLSTM single model classifier.
&ensp;Note:You can switch the train/test mode using the "train" flag.  
  
SPINN  
&ensp;Note: You may want to delete the zipped file in the data folder before running any code.  
&ensp;First, run the "saveEmbedding.py" code to reduce the size of the word embedding. Our code assumes that the original word embedding file is located in the folder "bioWordEmbedding" and the shortened word embedding file is located inside the "shorten_bc6" folder.  
&ensp;Second, run the "spinn_chemprot.py" code for the SPINN single model classifier.  
  
## Extra information:  
&ensp;Potision Embedding implementation - treeLSTM only  
&ensp;The relative distance value in the original position embedding has a range from -21 to 21. When the absolute distance value is greater than 5, same vector is given in units of 5.  
&ensp;In the preprocessed test data, the position embedding range is from 0 to 18. First, we merge the absolute distance values that share the same vectors, then the range is changed into -9 to 9. Secondly, we added 9 to the each distance value, because we do not want unnecessary negative values in the tree nodes.  

&ensp;This code do not contain automatic ensemble because it requires human hand for now. However, it is easy to implement.

## Data Format:  
tree-LSTM  
&ensp;After preprocessing step, we separate each pairs into lines. Each line contains
+ PMID,
+ preprocessed sentence (words are separated with comma.),
+ Interaction,
+ BioCreative6 type(if negative, ""),
+ the first target entity id,
+ the first target entity name,
+ the second target  entity id,
+ the second target entity name,
+ parsed tree of a sentence,
+ and words in a sentence separated with comma.  
Each element is separated with "\t".

Every node in a tree has the follwing input format.
* (label/Fsc/Fd1/Fd2 content)
The label is the answer label of a entity pair, and all the nodes of a tree share the same label as the root node. The model has six classes for classification. The Fsc/Fd1/Fd2 are the subtree containment (context), position1, and position2 features of a node, respectively. A node in a tree could be a leaf or an internal node. The content in the node is an input word if it is a leaf node, and if the current node is not a leaf node, the contents consists of children nodes of the current node.

SPINN  
&ensp;After preprocessing step, we separate each pairs into lines. Each line contains
+ label (0 to 5),
+ parsed tree of a sentence,
+ preprocessed sentence (words are separated with comma.),
+ PMID,
+ the first target entity id,
+ the second target  entity id,  
Each element is separated with "\t".  
