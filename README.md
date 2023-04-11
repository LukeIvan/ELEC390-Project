
# ELEC 390 Principles of Design & Development Final Project
The goal of this project is to build a desktop application that can distinguish between ‘walking’ and ‘jumping’ with reasonable accuracy, using the data collected from the accelerometers of a smartphone.

## Project Description
The project involves building a small and simple desktop application that accepts accelerometer data (x, y, and z axes) in CSV format, and writes the outputs into a separate CSV file. The output CSV file contains the labels (‘walking’ or ‘jumping’) for the corresponding input data. For classification purposes, the system will use a simple classifier, i.e., logistic regression.

In order to accomplish the goal of the final project and complete the report, the following 7 steps are required:

1. Data collection
2. Data storing
3. Visualization
4. Pre-processing
5. Feature extraction
6. Training the model
7. Creating a simple desktop application with a simple UI that shows the output
## Step 1. Data collection
In this step, we collected data using our smartphone while ‘walking’ and ‘jumping’. We used the Phyphox app to collect accelerometer data in CSV format.

Data collection protocol:

-Each team member participated in the data collection process to create a total of 3 subsets (1 -per member).
-The phone was placed in different positions to maximize diversity.
-The duration of data collection by each member exceeded 5 minutes.
-The dataset was roughly balanced.
## Step 2. Data storing
Once the data was collected, it was stored in a CSV format. Each CSV file contained the accelerometer data (x, y, and z axes), along with the corresponding label (‘walking’ or ‘jumping’).

## Step 3. Visualization
In this step, we visualized the collected data using Matplotlib to get a better understanding of it.

## Step 4. Pre-processing
In this step, we pre-processed the data to prepare it for feature extraction and training. We removed outliers, scaled the data, and split it into training and testing sets.

## Step 5. Feature extraction
In this step, we extracted features from the pre-processed data using NumPy. We extracted features such as mean, variance, skewness, and kurtosis that were relevant to the task of distinguishing between ‘walking’ and ‘jumping’.

## Step 6. Training the model
In this step, we trained the logistic regression model using scikit-learn. We evaluated its performance using metrics such as accuracy, precision, and recall.

## Step 7. Creating a simple desktop application with a simple UI that shows the output
In this step, we created a desktop application that accepts accelerometer data in CSV format and outputs the corresponding labels (‘walking’ or ‘jumping’). The application uses the trained logistic regression model to classify the data. The application also has a simple user interface that allows the user to select a CSV file and view the output.

