#RF& SVM Implimentation for the Bachelor thesis Word sense Disambiguation in Low resource languages by Ines ataknit

# Libraries Importation
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import os

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Function to Load and Preprocess Data
def load_and_preprocess_data(train_file, test_file):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Assuming columns are named 'Text' for the sentences and 'Sense' for the labels
    train_df = train_df.dropna(subset=['Text'])
    test_df = test_df.dropna(subset=['Text'])
    
    # Encode the labels
    label_encoder = LabelEncoder()
    train_df['Sense'] = label_encoder.fit_transform(train_df['Sense'])
    test_df['Sense'] = label_encoder.transform(test_df['Sense'])

    # Vectorize the text data
    vectorizer = CountVectorizer(stop_words='english')
    X_train = vectorizer.fit_transform(train_df['Text'])
    X_test = vectorizer.transform(test_df['Text'])

    y_train = train_df['Sense']
    y_test = test_df['Sense']

    return X_train, y_train, X_test, y_test, label_encoder, train_df, test_df


# Function to calculate the majority class for the training data
def calculate_majority_class(train_data):
    # Find the most frequent class (sense) in the training data
    majority_class = train_data['Sense'].value_counts().idxmax()
    return majority_class

# Function to predict the majority class for every instance in the test set
def predict_majority_class(test_data, majority_class):
    # Predict the majority class for all test instances
    return [majority_class] * len(test_data)

# Function to evaluate the majority baseline
def evaluate_majority_baseline(train_data, test_data):
    # Calculate the majority class from the training data
    majority_class = calculate_majority_class(train_data)
    
    # Predict the majority class for the test data
    test_predictions = predict_majority_class(test_data, majority_class)
    
    # True labels from the test data
    y_test = test_data['Sense']
    
    # Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(y_test, test_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_predictions, average='weighted')
    # Return the evaluation metrics
    return accuracy, precision, recall, f1
    

# Function to train the Random Forest model
def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

# Function to train the SVM model
def train_svm(X_train, y_train):
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    return svm_model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    classification_rep = classification_report(y_test, y_pred)
    return accuracy, precision, recall, f1, classification_rep



# List of training and testing files
train_files = ['1 Pass (پاس).csv', '10 Zabaan (زبان).csv', '11 Baras (برس).csv', '12 Mulk (ملک).csv', '13 Car (کار).csv', '14 Rang (رنگ).csv', '15 Shakal (شکل).csv',
               '16 Sawal (سوال).csv', '17 Umar (عمر).csv', '18 Khoon (خون).csv', '19 Shukar (شکر).csv', '2 Daur (دور).csv', '20 Qisam (قسم).csv', '21 Zikar (ذکر).csv',
               '22 Darmiyan (درمیان).csv', '23 Deen (دین).csv', '24 Hal (حل).csv', '25 Bijli (بجلی).csv', '26 Tor (طور).csv', '27 Doctor (ڈاکٹر).csv', '28 Safar (سفر).csv',
               '29 Aayat (آیت).csv', '3 Pani (پانی).csv', '30 Khat (خط).csv', '31 Kehna (کہنا).csv', '32 Dekh (دیکھ).csv', '33 Mil (مل).csv', '34 Lag (لگ).csv',
               '35 Chal (چل).csv', '36 Soch (سوچ).csv', '37 Bhool (بھول).csv', '38 Parh (پڑھ).csv', '39 Samajh (سمجھ).csv', '4 Sir (سر).csv', '40 Khas (خاص).csv',
               '41 Tayyar (تیار).csv', '42 Band (بند).csv', '43 Zindah (زندہ).csv', '44 Mukammal (مکمل).csv', '45 Sahih (صحیح).csv', '46 Shareef (شریف).csv',
               '47 Kam (کم).csv', '48 Sahaamil (شامل).csv', '49 Ghair (غیر).csv', '5 Hissa (حصہ).csv', '50 Ahem (اہم).csv', '6 Roshni (روشنی).csv', '7 Dil (دل).csv', '8 Nazar (نظر).csv', '9 Kitaab (کتاب).csv']

test_files = ['1 Pass (پاس)-n.csv', '10 Zabaan (زبان)-n.csv', '11 Baras (برس)-n.csv', '12 Mulk (ملک)-n.csv', '13 Car (کار)-n.csv', '14 Rang (رنگ)-n.csv',
              '15 Shakal (شکل)-n.csv', '16 Sawal (سوال)-n.csv', '17 Umar (عمر)-n.csv', '18 Khoon (خون)-n.csv', '19 Shukar (شکر)-n.csv',
              '2 Daur (دور)-n.csv', '20 Qisam (قسم)-n.csv', '21 Zikar (ذکر)-n.csv', '22 Darmiyan (درمیان)-n.csv', '23 Deen (دین)-n.csv',
              '24 Hal (حل)-n.csv', '25 Bijili (بجلی)-n.csv', '26 Tor (طور)-n.csv', '27 Doctor (ڈاکٹر)-n.csv', '28 Safar (سفر)-n.csv',
              '29 Aayat (آیت)-n.csv', '3 Pani (پانی)-n.csv', '30 Khat (خط)-n.csv', '31 Kehna (کہنا)-v.csv', '32 Dekh (دیکھ)-v.csv',
              '33 Mil (مل)-v.csv', '34 Lag (لگ)-v.csv', '35 Chal (چل)-v.csv', '36 Soch (سوچ)-v.csv', '37 Bhool (بھول)-v.csv',
              '38 Parh (پڑھ)-v.csv', '39 Samajh (سمجھ)-v.csv', '4 Sir (سر)-n.csv', '40 Khas (خاص)-a.csv', '41 Tayyar (تیار)-a.csv',
              '42 Band (بند)-a.csv', '43 Zindah (زندہ)-a.csv', '44 Mukammal (مکمل)-a.csv', '45 Sahih (صحیح)-a.csv', '46 Shareef (شریف)-a.csv',
              '47 Kam (کم)-a.csv', '48 Sahaamil (شامل)-a.csv', '49 Ghair (غیر)-a.csv', '5 Hissa (حصہ)-n.csv', '50 Ahem (اہم)-a.csv',
              '6 Roshni (روشنی)-n.csv', '7 Dil (دل)-n.csv', '8 Nazar (نظر)-n.csv', '9 Kitaab (کتاب)-n.csv']


# Initialize an empty list to store metrics dataframes
all_metrics_dfs = []

# Main loop for training and evaluation on multiple files
for train_file, test_file in zip(train_files, test_files):
    print(f"Processing {train_file} and {test_file}")

    # Load and preprocess data
    X_train, y_train, X_test, y_test, label_encoder, train_df, test_df = load_and_preprocess_data(train_file, test_file)

    # Evaluate Majority Baseline using the original train_df and test_df
    majority_accuracy, majority_precision, majority_recall, majority_f1 = evaluate_majority_baseline(train_df, test_df)
    
    # Train and evaluate Random Forest
    rf_model = train_random_forest(X_train, y_train)
    # Capture all five values, including the classification report
    rf_accuracy, rf_precision, rf_recall, rf_f1, rf_classification_rep = evaluate_model(rf_model, X_test, y_test)

    # Train and evaluate SVM
    svm_model = train_svm(X_train, y_train)
    # Capture all five values, including the classification report
    svm_accuracy, svm_precision, svm_recall, svm_f1, svm_classification_rep = evaluate_model(svm_model, X_test, y_test)

    # Create metrics dictionary (you can ignore the classification report here)
    metrics = {
        'Word': train_file.split('_')[-1].split('.')[0],
        'Majority_accuracy': majority_accuracy,
        'Majority_precision': majority_precision,
        'Majority_recall': majority_recall,
        'Majority_f1': majority_f1,
        'RF_accuracy': rf_accuracy,
        'RF_precision': rf_precision,
        'RF_recall': rf_recall,
        'RF_f1': rf_f1,
        'SVM_accuracy': svm_accuracy,
        'SVM_precision': svm_precision,
        'SVM_recall': svm_recall,
        'SVM_f1': svm_f1,
    }
# Append the metrics dictionary to the list
all_metrics_dfs.append(pd.DataFrame([metrics]))

# Concatenate all metrics dataframes into a single dataframe
all_metrics_df = pd.concat(all_metrics_dfs, ignore_index=True)

# Save the concatenated dataframe to a CSV file
all_metrics_df.to_csv("evaluation_metrics_all.csv", index=False)

# Calculate and print the average of all metrics for each model
majority_avg_metrics = all_metrics_df[['Majority_accuracy', 'Majority_precision', 'Majority_recall', 'Majority_f1']].mean()
rf_avg_metrics = all_metrics_df[['RF_accuracy', 'RF_precision', 'RF_recall', 'RF_f1']].mean()
svm_avg_metrics = all_metrics_df[['SVM_accuracy', 'SVM_precision', 'SVM_recall', 'SVM_f1']].mean()

print("Random Forest Averages")
print("Average Accuracy: {:.7f}".format(rf_avg_metrics['RF_accuracy']))
print("Average Precision: {:.7f}".format(rf_avg_metrics['RF_precision']))
print("Average Recall: {:.7f}".format(rf_avg_metrics['RF_recall']))
print("Average F1 Score: {:.7f}".format(rf_avg_metrics['RF_f1']))

print("\nSVM Averages")
print("Average Accuracy: {:.7f}".format(svm_avg_metrics['SVM_accuracy']))
print("Average Precision: {:.7f}".format(svm_avg_metrics['SVM_precision']))
print("Average Recall: {:.7f}".format(svm_avg_metrics['SVM_recall']))
print("Average F1 Score: {:.7f}".format(svm_avg_metrics['SVM_f1']))

print("\nMajority Baseline Averages")
print("Average Accuracy: {:.7f}".format(majority_avg_metrics['Majority_accuracy']))
print("Average Precision: {:.7f}".format(majority_avg_metrics['Majority_precision']))
print("Average Recall: {:.7f}".format(majority_avg_metrics['Majority_recall']))
print("Average F1 Score: {:.7f}".format(majority_avg_metrics['Majority_f1']))

print("All metrics saved.")


#------------------------------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# Load the evaluation metrics
metrics_df = pd.read_csv('evaluation_metrics_all.csv')

# Calculate average metrics for both models
avg_metrics = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
    'Majority Baseline': [
        all_metrics_df['Majority_accuracy'].mean(),
        all_metrics_df['Majority_precision'].mean(),
        all_metrics_df['Majority_recall'].mean(),
        all_metrics_df['Majority_f1'].mean()
    ],

    'Random Forest': [
        metrics_df['RF_accuracy'].mean(),
        metrics_df['RF_precision'].mean(),
        metrics_df['RF_recall'].mean(),
        metrics_df['RF_f1'].mean()
    ],
    'SVM': [
        metrics_df['SVM_accuracy'].mean(),
        metrics_df['SVM_precision'].mean(),
        metrics_df['SVM_recall'].mean(),
        metrics_df['SVM_f1'].mean()
    ]
})

# 1. Bar Plot to Compare Average Metrics
plt.figure(figsize=(10, 6))
avg_metrics_melted = avg_metrics.melt(id_vars='Metric', value_vars=['Majority Baseline', 'Random Forest', 'SVM'])
sns.barplot(x='Metric', y='value', hue='variable', data=avg_metrics_melted)
plt.title('Average Performance Metrics: Majority Baseline vs Random Forest vs SVM')
plt.ylabel('Average Score')
plt.xlabel('Metric')
plt.ylim(0, 1) 
