# Bayes Classifier for email spam filter

Put your sample email into `test.txt` file and run `python spamham.py`.

Data is trained on a small subset of the [Enron](https://www.cs.cmu.edu/~enron/) dataset.

Laplace Smoothing is used to handle words that aren't present in the training set.

Classifier performance: 
* Accuracy: 0.9749
* Precision: 0.9819
* Recall: 0.9283
* F1 Score: 0.9544

Requires python 3+.
