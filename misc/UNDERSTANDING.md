### BalancedVsRandom 
- We will create two subsets (of the same size) of our binary dataset, one completely random, and one maintaining balance between classes. We will train then an SVM model and we will evaluate on the rest of our initial dataset. We will do the same for the external dataset.
- The results propose that better performance has the **balanced** subset. On both the rest of ETHOS Data, and the external data (on the hate class, which is the minority class as well in this dataset). 

### Generalizing-Binary-Experiment-D1
- Loading ETHOS and the dataset D1: Davidson, Thomas, et al. "Automated hate speech detection and the problem of offensive language." Proceedings of the International AAAI Conference on Web and Social Media. Vol. 11. No. 1. 2017.
- Finally, it would be interesting to investigate the overall performance of an SVMmodel trained on a combination dataset of those two
- Combined dataset gave better results.

### setA : DONE
- In this set of experiments we will try logistic regression, svms, ridge, decision trees, naive bayes and random forests classifiers across a wide variety of parameters for each algorithm and test them via nested cross validation method.

### setA+XHS
- DUNNO THE DIFFERENCE

### setA-on-external
- Like setA but on external dataset

### setB : DONE
- In this set of experiments we will try AdaBoost, GraBoost and Bagging across a wide variety of parameters for each algorithm and test them via nested cross validation method.

### setC
- Testing a variety of NN architectures with Embeddings
- FastText and Glove

### setD
- BERT Tests
- Better to run on GPU(Google Colab)
- Uses bert_text library

### setE
- In this set of experiments we will try classic multi label algorithms to find the best classifiers.

### setZ
- Classic Multi-label algorithms + Neural Networks and Embeddings