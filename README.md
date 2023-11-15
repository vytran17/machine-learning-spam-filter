# Bayes Classifier for email spam filter

Put your sample email into `test.txt` file and run `python spamham.py`.

Data is trained on a small subset of the [Enron](https://www.cs.cmu.edu/~enron/) dataset.

Laplace Smoothing is used to handle words that aren't present in the training set. Laplace smoothing is a simple but effective method to handle zero probabilities in
statistical models. 

Requires python 3+.

**The preprocessing steps:**
  * Removed “Subject: “ prefix
  * Make it lowercase
  * Removed everything that’s not a thru z
  * Remove repeated spaces
  * Removed English stop words based on nltk library

**The methodology used to calculate the probabilities**

Bayesian Theory: The fundamental idea behind a Bayesian spam filter is Bayes'
theorem, which gives us a way to find the probability of an event occurring given prior knowledge.

**a. Training the Classifier:**
For every token in our vocabulary, we calculated the probability of that token appearing
in a spam email and the probability of it appearing in a non-spam email.

**b. Classification of New Emails:**
For a new email, multiply together the spam probabilities of each of its tokens and then
do the same for the non-spam probabilities. Apply Bayes' theorem to determine if it's
more likely to be spam or not.

**The performance metrics of the classifier**

**Accuracy: 0.9749** - This measures the proportion of true results (both true positives and true negatives) among the total number of cases examined. An accuracy of 0.9749 means that 97.49% of the predictions made by the classifier are correct.

**Precision: 0.9819** - Precision is about being precise, i.e., how accurate the model's predictions are. It is the ratio of true positives (correct positive predictions) to the total number of positive predictions made (both true positives and false positives). A precision of 0.9819 indicates that when the model predicts a class, it is correct 98.19% of the time.

**Recall: 0.9283** - Also known as sensitivity, recall measures the ability of the classifier to find all the relevant cases within a dataset. It is the ratio of true positives to the sum of true positives and false negatives. A recall of 0.9283 means the model correctly identifies 92.83% of all actual positive cases.

**F1 Score: 0.9544** - The F1 Score is a measure of a test's accuracy that considers both the precision and the recall. It is the harmonic mean of precision and recall, providing a balance between them. An F1 Score of 0.9544 suggests that the classifier has a high precision and recall balance.
