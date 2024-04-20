from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog,filedialog
import tkinter
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

pd.options.mode.chained_assignment = None

# Tkinter UI
main = Tk()
main.title("Sleep Disorder Prediction")
main.geometry("1300x1200")

global dataset
global X, y, X_train, X_test, y_train, y_test, clf, text

def uploadDataset():
    global dataset
    file_path = filedialog.askopenfilename(title="Select Dataset File", filetypes=[("CSV files", "*.csv")])
    if file_path:
        dataset = pd.read_csv(file_path)
        text.insert(END, 'Dataset loaded\n')
        text.insert(END, 'Sample dataset\n' + str(dataset.head()) + "\n\n\n")

def preprocessData():
    global X, y, text
    text.delete('1.0', END)

    # Assuming label is 'Sleep Disorder'
    label_encoder = LabelEncoder()
    dataset['Sleep Disorder_encoded'] = label_encoder.fit_transform(dataset['Sleep Disorder'])

    # Define features and target variable
    X = dataset.drop(['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder', 'Blood Pressure'], axis=1)
    y = dataset['Sleep Disorder_encoded'].values

    text.insert(END, "Data preprocessed\n")
    text.insert(END, 'dataset before label encoding\n' + str(dataset.head()) + "\n\n\n")
    text.insert(END, 'dataset after label encoding\n' + str(X.head()) + "\n\n\n")

def performance_evaluation(model_name, y_true, y_pred, classes):
    accuracy = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average='micro') 
    rec = recall_score(y_true, y_pred, average='micro') 
    f1s = f1_score(y_true, y_pred, average='micro')  
    report = classification_report(y_true, y_pred, target_names=classes)

    text.insert(END, f"{model_name} Accuracy: {accuracy}\n\n")
    text.insert(END, f"{model_name} Precision: {pre}\n\n")
    text.insert(END, f"{model_name} Recall: {rec}\n\n")
    text.insert(END, f"{model_name} F1-score: {f1s}\n\n")
    text.insert(END, f"{model_name} Classification report\n{report}\n\n")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

def trainRandomForestModel():
    global X, y, X_train, X_test, y_train, y_test, clf, text
    text.delete('1.0', END)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Random Forest classifier
    clf = RandomForestClassifier()

    # Train the model on the training data
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    classes=['Normal','Sleep Apena','Insomania']
    text.insert(END, "Random Forest Model trained\n")
    performance_evaluation("Random Forest Model", y_test, y_pred, classes)

def trainDecisionTreeModel():
    global X, y, X_train, X_test, y_train, y_test, clf, text
    text.delete('1.0', END)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Decision Tree classifier
    clf = DecisionTreeClassifier()

    # Train the model on the training data
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    classes=['Normal','Sleep Apena','Insomania']
    text.insert(END, "DTC trained\n")
    performance_evaluation("DTC", y_test, y_pred, classes)

def predictSleepDisorder():
    global clf, text

    # Assuming you have a filedialog to get the test dataset
    file_path = filedialog.askopenfilename(title="Select Test Dataset File", filetypes=[("CSV files", "*.csv")])
    test_data = pd.read_csv(file_path)
    label_encoder = LabelEncoder()
    dataset['Sleep Disorder_encoded'] = label_encoder.fit_transform(dataset['Sleep Disorder'])
    # Assuming label is 'Sleep Disorder'
    test_data['Sleep Disorder_encoded'] = label_encoder.transform(test_data['Sleep Disorder'])

    # Extract features for prediction
    X_test_predict = test_data.drop(['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder', 'Blood Pressure'], axis=1)

    # Make predictions on the test data
    predictions = clf.predict(X_test_predict)

    # Print predictions to the Tkinter Text widget
    text.insert(END, "Predictions for Sleep Disorder:\n")
    for i, prediction in enumerate(predictions):
        sample_data = X_test_predict.iloc[i]
        formatted_data = ', '.join(f"{col}: {sample_data[col]}" for col in X_test_predict.columns)
        text.insert(END, f"Features: {formatted_data}\n")
        text.insert(END, f"Test Data {i+1}: {label_encoder.inverse_transform([prediction])[0]}\n\n\n")

font = ('times', 18, 'bold')
title = Label(main, text='Predictive Modeling Approach for Sleep Disorder using Sleep Health and Lifestyle Properties', justify=LEFT)
title.config(bg='white', fg='red')   
title.config(font=font)           
title.config(height=3, width=120)       
title.pack()

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=20, y=100)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocess Data", command=preprocessData)
preprocessButton.place(x=20, y=150)
preprocessButton.config(font=font1)

trainRFCButton = Button(main, text="Train Random Forest Model", command=trainRandomForestModel)
trainRFCButton.place(x=20, y=200)
trainRFCButton.config(font=font1)

trainDTButton = Button(main, text="Train Decision Tree Model", command=trainDecisionTreeModel)
trainDTButton.place(x=20, y=250)
trainDTButton.config(font=font1)

predictButton = Button(main, text="Predict Sleep Disorder", command=predictSleepDisorder)
predictButton.place(x=20, y=300)
predictButton.config(font=font1)

text = Text(main, height=30, width=85)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=500, y=100)
text.config(font=font1)

main.config(bg='white')  # Change background color of the main window
main.mainloop()
