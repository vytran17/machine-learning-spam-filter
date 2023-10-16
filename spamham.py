#mathematical functions for ex pow & log
import math  
#read CSV, Excel, SQL databases
import pandas as pd       
#string operations related to pattern matching using regular expressions                            
import re 

#accumulating data, such as counting occurrences or grouping by some criteria
from collections import defaultdict                   
#turning a text string into a list of words and punctuation - text analysis or modeling in NLP
from nltk.tokenize import word_tokenize    
#filtered out stop words before processing natural language data           
from nltk.corpus import stopwords
#scikit-learn library for ML in Python - used to split a dataset into two parts: a training set and a test set
from sklearn.model_selection import train_test_split  

#CLEAN EMAIL
def clean_email(emailbody):
    # 1 - Remove the "Subject:" prefix if present
    emailbody = emailbody.replace("Subject:", "", 1).strip()

    # 2 - Keep only alphabetical characters (removing digits, punctuation, etc.).
    emailbody = re.sub('[^a-zA-Z]', ' ', emailbody)
    # 3 - Remove extra spaces
    emailbody = re.sub('\s+', ' ', emailbody)

    # 4 - Tokenize the email body into words
    tokens = word_tokenize(emailbody.lower())
    # 5 - Filter out stop words
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # 6 - return a sest of the tokens
    return set(filtered_tokens)  

# LOAD STOP WORDS
stop_words = set(stopwords.words('english'))

# LOAD CSV FILE INTO A DATAFRAME
df = pd.read_csv('spam_ham_dataset.csv')

# SPLIT DATA INTO TRAINING (80%) AND TESTING (20%) SETS
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42) #42 is just an integer used to seed the random number generator

#TRAINING

# Initialize counters
spam_word_counts = defaultdict(int)
ham_word_counts = defaultdict(int)
total_spam_emails = 0
total_ham_emails = 0

# Process training data to compute word probabilities
for _, row in train_df.iterrows():
    emailbody = row['text']
    spamorham = row['label']
    unique_tokens = clean_email(emailbody)
    
    if spamorham == 'spam':
        total_spam_emails += 1
        for token in unique_tokens:
            spam_word_counts[token] += 1
    elif spamorham == 'ham':
        total_ham_emails += 1
        for token in unique_tokens:
            ham_word_counts[token] += 1

# Calculate probabilities
total_spam_words = sum(spam_word_counts.values())
total_ham_words = sum(ham_word_counts.values())

p_spam = total_spam_emails / (total_spam_emails + total_ham_emails)
p_ham = total_ham_emails / (total_spam_emails + total_ham_emails)

# Sort word counts in descending order and get the top 25
sorted_spam_word_counts = dict(sorted(spam_word_counts.items(), key=lambda item: item[1], reverse=True))
sorted_ham_word_counts = dict(sorted(ham_word_counts.items(), key=lambda item: item[1], reverse=True))
top_25_spam = {k: sorted_spam_word_counts[k] for k in list(sorted_spam_word_counts)[:25]}
top_25_ham = {k: sorted_ham_word_counts[k] for k in list(sorted_ham_word_counts)[:25]}

# Print the top 25 words
print(f"Top 25 spam words: {top_25_spam} \n")
print(f"Top 25 ham words: {top_25_ham} \n")
print(f"Spams: {p_spam} hams: {p_ham}\n")

# LEts test some emails.
def classify_email(emailtext):
    email_tokens = clean_email(emailtext)

    # For each test email, multiply probabilities of each word using spam and ham probabilities from train set
    log_spam_likelihood = 0
    log_ham_likelihood = 0

    for token in email_tokens:
        # Use Laplace smoothing
        log_spam_likelihood += math.log((spam_word_counts.get(token, 0) + 1) / (total_spam_words + len(spam_word_counts)))
        log_ham_likelihood += math.log((ham_word_counts.get(token, 0) + 1) / (total_ham_words + len(ham_word_counts)))

    # Include the prior probabilities
    log_spam_likelihood += math.log(p_spam)
    log_ham_likelihood += math.log(p_ham)

    return 'spam' if log_spam_likelihood > log_ham_likelihood else 'ham'

TP = 0
FP = 0
FN = 0
TN = 0

for _, row in test_df.iterrows():
    true_label = row['label']
    predicted_label = classify_email(row['text'])

    # Check if we got it right and update counters
    if true_label == 'spam' and predicted_label == 'spam':
        TP += 1
    elif true_label == 'ham' and predicted_label == 'spam':
        FP += 1
    elif true_label == 'spam' and predicted_label == 'ham':
        FN += 1
    elif true_label == 'ham' and predicted_label == 'ham':
        TN += 1

accuracy = (TP + TN) / len(test_df)
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# Now let's test user provided email!
with open('test.txt', 'r') as file:
    test_email = file.read()

predicted_label_for_test_txt = classify_email(test_email)

print(f"\nThe email in test.txt is predicted as: {predicted_label_for_test_txt.upper()}\n")



