# ID3 & KNNForest algorithms implementation
AI - ID3 algo for building decision trees
a small project of ML programming.
2 data files included, train and test.

study the training file of a few hundred subjects with 20+ features of physical info, each subject has a diagnosis of Coronavirus infection (B)-healthy or (M)-infected.
the program studies the train test and creates a classifier which diagnoses each subj from the test group.

ID3 - basic ID3 algorithm implementation with a dynamic partition of continuous features, find the best feature and best threshold for biggest information gain, 
      one decision tree is built based on train.csv file reaches 96% accuracy on test.csv file prediction.
      
CostSensitiveID3 - adjustments to lower the false negative even at the expense of false postive (each false negative worth 10 false pasitive), accuracy calculated
                   as follows: (0.1 * false_positive + false_negative) / test_set_size.
                   
KNNForest - different kind of classifier: divide the training group into N(parameter) groups, build the tree based on each group, build "centroid", a vector of avg value of features of all subjects in the group, classify each subject with K(parameter) trees based on Euclid dist of the centroid from the subject and return the majority decision of all trees. reaches 100% accuracy on the test group.
            
ImprovedKNNForest - normalize all features to decide Euclid distance.
