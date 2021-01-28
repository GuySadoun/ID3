# ID3
AI - ID3 algo for building decision trees
small project of ML programming.
2 data files included, train and test.

study the train file of few hundred subjects with 20+ feature of physical info, each subject have diagnosis of Corona virus infection (B)-healthy or (M)-infected.
the program study the train test and create classifier which diagnos each subj from test group.

ID3 - basic ID3 algorithm implementation with dynamic partition of continuous features, find the best feature and best threshold for biggest information gain, 
      one decision tree is builed based on train.csv file reaches 96% accuracy on test.csv file prediction.
      
CostSensitiveID3 - adjustments to lower the false negative even at the expense of false postive (each false negative worth 10 false pasitive), accuracy calculated
                   as follows: (0.1 * false_positive + false_negative) / test_set_size.
                   
KNNForest - different kind of classifier: devide the train group to N(parameter) groups, build tree based on each group, build "centroid", a vector of avg value 
            of features of all subjects in group, classify each subject with K(parameter) trees baset on euclid dist of centroid from subject and return the
            majority decision of all trees. reaches 100% accuracy on test group.
            
ImprovedKNNForest - normalize all features to decide euclid distance.
