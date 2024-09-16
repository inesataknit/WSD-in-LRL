# **Word Sense Disambiguation (WSD) Using Random Forest and SVM**

### **Project Overview**
This project applies two supervised machine learning models, **Random Forest** and **Support Vector Machine (SVM)**, to the task of **Word Sense Disambiguation (WSD)** for the low-resource Urdu language. The study compares the performance of these models with a **Majority Baseline** to evaluate their effectiveness in handling the complexities of WSD in imbalanced datasets.

### **Models Implemented**
1. **Random Forest**: A robust ensemble model that is used to predict word senses based on contextual features.
2. **Support Vector Machine (SVM)**: A high-precision model that balances precision and recall, especially useful in scenarios where false positives are costly.
3. **Majority Baseline**: A simple reference model that always predicts the most frequent word sense from the training data.

### **Project Structure**
- **/src**: Contains the Python code for loading the dataset, training the models, and evaluating their performance.
  - **`WSD_Implementation.py`**: Main implementation file that trains Random Forest, SVM, and evaluates against the Majority Baseline.
- **/evaluation_metrics_all.csv**: Contains the evaluation results and metrics table.
- **/Figure.png**: Visualizations comparing model performance across accuracy, precision, recall, and F1-score.


