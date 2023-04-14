# LIBRARIES USED FOR DATA STORING
import pandas as pd
import numpy as np
import h5py

# LIBRARY USED FOR VISUALIZATION
import matplotlib.pyplot as plt

# LIBRARIES USED FOR PREPROCESING
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# LIBRARY FEATURE EXTRACTION
from scipy.stats import skew, kurtosis

# LIBRARIES FOR TRAINING MODEL AN DEVALUATING PERFORMANCE
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report

# LIBRARIES FOR DESKTOP APPLICATION
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
# import csv




# DATA STORING

def segment_and_split_data(csv_path, window_size, train_frac):
    # read in the CSV file
    df = pd.read_csv(csv_path)

    # segment the data into 5-second windows
    segments = []
    for i in range(0, len(df)-window_size+1, window_size):
        segment = df[i:i+window_size]
        segments.append(segment)
    segments = np.array(segments)

    # split the data into training and test sets
    train_size = int(len(segments) * train_frac)
    train_segments = segments[:train_size]
    test_segments = segments[train_size:]

    # shuffle the training and test sets
    np.random.shuffle(train_segments)
    np.random.shuffle(test_segments)

    return train_segments, test_segments

# Import walking data  - Segmenting data into non-overlapping widows, each contianing 5 secons of data (200 data samples collected per second)
#                      - Segments used to train ML model on fixed-size time windowns of the input data
member1_walking_train, member1_walking_test = segment_and_split_data(os.path.join('Bryce_Data', 'walking.csv'), window_size=5*200, train_frac=0.9)
member2_walking_train, member2_walking_test = segment_and_split_data(os.path.join('Luke_Data', 'walking.csv'), window_size=5*200, train_frac=0.9)
member3_walking_train, member3_walking_test = segment_and_split_data(os.path.join('Ryan_Data', 'walking.csv'), window_size=5*200, train_frac=0.9)

# Import jumping data
member1_jumping_train, member1_jumping_test = segment_and_split_data(os.path.join('Bryce_Data', 'jumping.csv'), window_size=5*200, train_frac=0.9)
member2_jumping_train, member2_jumping_test = segment_and_split_data(os.path.join('Luke_Data', 'jumping.csv'), window_size=5*200, train_frac=0.9)
member3_jumping_train, member3_jumping_test = segment_and_split_data(os.path.join('Ryan_Data', 'jumping.csv'), window_size=5*200, train_frac=0.9)

# Store the data in an HDF5 file
with h5py.File('ELEC390FinalProject.h5', 'w') as hf:
    # create the groups for the members' datasets
    member1_grp = hf.create_group('Member1 - Bryce')
    member2_grp = hf.create_group('Member2 - Ryan')
    member3_grp = hf.create_group('Member3 - Luke')

    # Store the walking and jumping data for each member
    member1_grp.create_dataset('walking_data', data=pd.read_csv(os.path.join('Bryce_Data', 'walking.csv')))
    member1_grp.create_dataset('jumping_data', data=pd.read_csv(os.path.join('Bryce_Data', 'jumping.csv')))
    member2_grp.create_dataset('walking_data', data=pd.read_csv(os.path.join('Luke_Data', 'walking.csv')))
    member2_grp.create_dataset('jumping_data', data=pd.read_csv(os.path.join('Luke_Data', 'jumping.csv')))
    member3_grp.create_dataset('walking_data', data=pd.read_csv(os.path.join('Ryan_Data', 'walking.csv')))
    member3_grp.create_dataset('jumping_data', data=pd.read_csv(os.path.join('Ryan_Data', 'jumping.csv')))

    # Create the Dataset Group with train and test subgroups
    dataset_grp = hf.create_group('dataset')
    train_grp = dataset_grp.create_group('train')
    test_grp = dataset_grp.create_group('test')

    # Store the walking data (0) and jumping data (1) for train and test sets
    train_data = np.vstack((member1_walking_train, member2_walking_train, member3_walking_train,
                            member1_jumping_train, member2_jumping_train, member3_jumping_train))
    train_labels = np.hstack((np.zeros(member1_walking_train.shape[0] + member2_walking_train.shape[0] + member3_walking_train.shape[0]),
                              np.ones(member1_jumping_train.shape[0] + member2_jumping_train.shape[0] + member3_jumping_train.shape[0])))
    test_data = np.vstack((member1_walking_test, member2_walking_test, member3_walking_test,
                           member1_jumping_test, member2_jumping_test, member3_jumping_test))
    test_labels = np.hstack((np.zeros(member1_walking_test.shape[0] + member2_walking_test.shape[0] + member3_walking_test.shape[0]),
                             np.ones(member1_jumping_test.shape[0] + member2_jumping_test.shape[0] + member3_jumping_test.shape[0])))

    # Shuffle train and test data
    train_permutation = np.random.permutation(train_data.shape[0])
    train_data = train_data[train_permutation]
    train_labels = train_labels[train_permutation]
    test_permutation = np.random.permutation(test_data.shape[0])
    test_data = test_data[test_permutation]
    test_labels = test_labels[test_permutation]

    # Store train and test data and labels
    train_grp.create_dataset('data', data=train_data)
    train_grp.create_dataset('labels', data=train_labels)
    test_grp.create_dataset('data', data=test_data)
    test_grp.create_dataset('labels', data=test_labels)




# VISUALIZATION

# Function that will be used for ploting walking and jumping data --- WORKS!!!
def plot_sample(sample, title):
    plt.figure()
    plt.plot(sample[:, 0], label='x-axis')
    plt.plot(sample[:, 1], label='y-axis')
    plt.plot(sample[:, 2], label='z-axis')
    plt.title(title)
    plt.xlabel('Time - (Samples)')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.show()

# Plot a walking sample
# plot_sample(member1_walking_train[0], "Member1 Walking Sample") #-- WORKS!!!

# Plot a jumping sample
# plot_sample(member1_jumping_train[0], "Member1 Jumping Sample") #-- WORKS!!!

# # Scatter plot of walking and jumping data -- WORKS!!!
# plt.figure()
# plt.scatter(member1_walking_train[:, :, 0].flatten(), member1_walking_train[:, :, 1].flatten(), alpha=0.5, label='Walking')
# plt.scatter(member1_jumping_train[:, :, 0].flatten(), member1_jumping_train[:, :, 1].flatten(), alpha=0.5, label='Jumping')
# plt.xlabel('X-Axis Acceleration')
# plt.ylabel('Y-Axis Acceleration')
# plt.legend()
# plt.title("Member1 Walking vs Jumping Scatter Plot")
# plt.show()




# PRE-PROCESSING

# Apply a moving average filter (low pass - could be on exam) to the data - removes noise
def moving_average_filter(data, window_size=3):
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[2]):
        filtered_data[:, :, i] = signal.convolve2d(data[:, :, i], np.ones((1, window_size))/window_size, mode='same', boundary='symm')
    return filtered_data


# FEATURE EXTRACTION
# Feature extraction - mean, standard deviation, and correlation performed for each axis
def extract_features(data):
    # Mean - avg. value of data points in each axis of the time window.
    mean = np.mean(data, axis=1)

    # Standard Deviation - measures dispersion of datapoints around the mean. Larger std = wider range of values
    std = np.std(data, axis=1)

    # Minimum - Smallest value in each axis of time window
    minimum = np.min(data, axis=1)

    # Maximum - Largest value in each axis of time window
    maximum = np.max(data, axis=1)

    # Range - difference between max and min valeus in each axis of time window 
    data_range = maximum - minimum

    # Median - middle value of the data poins in each axis of the time window (considering points are sorted)
    median = np.median(data, axis=1)

    # Variance - avg. of squared differences from mean for each axis of time window
    variance = np.var(data, axis=1)

    # Skewness - measures the asymmetry of data distribution for each axis of the time window. + = tail on right side, - = tail on left side
    skewness = skew(data, axis=1)

    # Kurtosis - measures the tailedness of data distribution for each axis of the time window. 
    # Larger value = more extreme distribution with more outliers. Smaller = more uniform distribution and less outliers
    kurt = kurtosis(data, axis=1)

    # k-order differences (differencs between consective data points if k = 1 or been k-1-order differences) used as input to calculate the difference mean values 
    first_diff = np.diff(data, axis=1)[:, :-1]
    second_diff = np.diff(first_diff, axis=1)[:, :-1]
    third_diff = np.diff(second_diff, axis=1)[:, :-1]
    fourth_diff = np.diff(third_diff, axis=1)[:, :-1]
    fifth_diff = np.diff(fourth_diff, axis=1)[:, :-1]
    sixth_diff = np.diff(fifth_diff, axis=1)[:, :-1]

    # First order difference mean - Avg of the first-order differences (differences between consecutive data points) in each axis of time window. 
    # Gives insight as to the local chagne in the signal 
    first_order_diff_mean = np.mean(first_diff, axis=1)

    # Second order difference mean - Avg of the second-order differences (differences between consecutive first-order differences) in each axis of time window. 
    # Gives insight as to the acceleration or deceleration of a signal
    second_order_diff_mean = np.mean(second_diff, axis=1)

    # Third order difference mean - Avg of the thrid-order differences (differences between consecutive second-order differences) in each axis of time window. 
    # Gives insight as to higher-order changes in a signal
    third_order_diff_mean = np.mean(third_diff, axis=1)
    fourth_order_diff_mean = np.mean(fourth_diff, axis=1)
    fifth_order_diff_mean = np.mean(fifth_diff, axis=1)
    sixth_order_diff_mean = np.mean(sixth_diff, axis=1)

    features = np.hstack((
        mean, std, minimum, maximum, data_range, median, variance, skewness, kurt,
        first_order_diff_mean, second_order_diff_mean, third_order_diff_mean,
        fourth_order_diff_mean, fifth_order_diff_mean, sixth_order_diff_mean
    ))
    
    return features


# Remove outliers - Isolation forest is unsupervised learning algo for anomaly detection and ultimatley isolation
def remove_outliers(features, labels, contamination=0.1):
    isolation_forest = IsolationForest(contamination=contamination)
    inliers = isolation_forest.fit_predict(features) == 1
    return features[inliers], labels[inliers]

# Normalize the data - creates a uniform range across our dataset since we care more about the shapes and patterns 
# in the data as opposed to the actual values given by the sensors (also may be on exam)
def normalize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# Preprocess the data - apply above funcitons to the test and training data moving average filter
window_size = 3
train_data_filtered = moving_average_filter(train_data, window_size)
test_data_filtered = moving_average_filter(test_data, window_size)

train_features = extract_features(train_data_filtered)
test_features = extract_features(test_data_filtered)

train_features, train_labels = remove_outliers(train_features, train_labels)
test_features, test_labels = remove_outliers(test_features, test_labels)

train_features_normalized = normalize_data(train_features)
test_features_normalized = normalize_data(test_features)




# CREATE A CLASSIFIER

# Train a logistic regression model to classify data into walking and jumping classes.
# Ensure trainign wiht CV (cross-validation) - assess ML Model to tune its hypeerparemeters (improves generalization and avoids overfitting)
# 5 folds for cross-validaiton process (healthy balance between computation time and reliability of the performance estimation)
# Random state to a fixed value of 42 ensures reproducible results - 42 is just the standard 
lr_model = LogisticRegressionCV(cv=5, random_state=42)
lr_model.fit(train_features_normalized, train_labels)

# Apply model on test set
test_predictions = lr_model.predict(test_features_normalized)

# Compute accuracy of the model
accuracy = accuracy_score(test_labels, test_predictions)

# Print the results from the accuracy and the classification reports
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(test_labels, test_predictions))

# Monitoring via Training Curve - Monitor the taraining of the logic regression model using LogisticRegressionCV -- WORKS!!!
# plt.figure()
# plt.plot(lr_model.Cs_, np.mean(lr_model.scores_[1], axis=0))
# plt.xscale('log')
# plt.xlabel('Regularization Parameter (C)')
# plt.ylabel('Mean Cross-Validated Score')
# plt.title('Training Curve for Logistic Regression')
# plt.show()





# DEPLOYING TRAINED CLASSIFIER IN A DESKTOP APP

# Create the main window
root = tk.Tk()
root.title("Physical Activity Classifier")

# Button to upload a CSV file
def upload_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        segments, predictions = process_input_file(file_path)

        # Save output from classifier to a CSV file
        output_df = pd.DataFrame(columns=["segment", "label"])
        for i, pred in enumerate(predictions):
            label = "walking" if pred == 0 else "jumping"
            output_df = pd.concat([output_df, pd.DataFrame({"segment": [i], "label": [label]})], ignore_index=True)


        output_file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if output_file_path:
            output_df.to_csv(output_file_path, index=False)

        # Create and display plot
        fig, ax = plt.subplots()
        walking_count = (predictions == 0).sum()
        jumping_count = (predictions == 1).sum()
        ax.bar(["Walking", "Jumping"], [walking_count, jumping_count])
        ax.set_title("Activity Classification")

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(row=3, column=0)

        messagebox.showinfo("Success", "CSV file processed and output saved.")


# Processing the input file
def process_input_file(file_path):
    window_size = 5*200
    train_frac = 0.9  # You may need to adjust this value depending on your use case

    # Use the existing segment_and_split_data function
    train_segments, test_segments = segment_and_split_data(file_path, window_size, train_frac)

    # Choose which data I can use for predictions (train or test)
    segments = train_segments  # or test_segments, depending on your what I want for use case

    # Continue with the rest of the function
    filtered_data = moving_average_filter(segments, window_size)
    features = extract_features(filtered_data)
    normalized_features = normalize_data(features)
    predictions = lr_model.predict(normalized_features)

    return segments, predictions

upload_btn = tk.Button(root, text="Upload CSV", command=upload_csv)
upload_btn.grid(row=0, column=0)

root.mainloop()
