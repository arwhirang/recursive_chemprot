# recursive_chemprot
This is the open source repository of my Recursive Neural Network model for the BioCreative 6 track5 - chemprot task.

## Requirements:  
&ensp;Tensorflow ver 1.0  (we do not test the other versions, we found that tensorflow fold is known to be working best at tf version 1.0.0.)  
&ensp;Tensorflow Fold https://github.com/tensorflow/fold  
&ensp;python 3.4 or later  
&ensp;Common libraries:  
&ensp;&ensp;gensim https://radimrehurek.com/gensim/install.html  
&ensp;&ensp;sklearn http://scikit-learn.org/stable/install.html  
&ensp;&ensp;numpy  
&ensp;&ensp;nltk  

## Data:  
&ensp;Use the preprocessed test, validation and train data in the Demo/data folder. This is not the original data.  
&ensp;Word embedding is not included. visit : http://evexdb.org/pmresources/vec-space-models/

## Training and Testing:  
&ensp;First, run the "saveWEUsedinDataOnly" code to reduce the size of the word embedding.  
&ensp;Second, run the "BC6train" code for the DDI detection single model classifier.
&ensp;Note : you may want to change the directory path for reserving logits.

&ensp;Potision Embedding implementation  
&ensp;The relative distance value in the original position embedding has a range from -21 to 21. When the absolute distance value is greater than 5, same vector is given in units of 5. 
&ensp;In the preprocessed test data, the position embedding range is from 0 to 18. First, we merge the absolute distance values that share the same vectors, then the range is changed into -9 to 9. Secondly, we added 9 to the each distance value, because we do not want unnecessary negative values in the tree nodes.  

&ensp;This code do not contain automatic ensemble because it requires human hand for now. However, it is easy to implement.

## Data Format:  
&ensp;After preprocessing step, we separate each pairs into lines. Each line contains
+ PMID,
+ preprocessed sentence,
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
