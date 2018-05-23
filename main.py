import Classify
import csv      # import the csv module to read the csv file
import random   # import the random module to shuffle the training data
import bs4      # import the bs4 module as to change the data from html to Text
import os       # Check whether File Exist or not
import warnings # Suppres the bs4 Warning
import math     # import math module for floor

# Suppress the UserWarning Warning caused by the Beautifull Soup Module
warnings.filterwarnings("ignore",category=UserWarning,module='bs4')

trainingData = list() # Contains The Training Data
testingData = list() # Contains The Testing Data

calssifier = Classify.SpamClassification() # Create The Classifier Object

nTestingData = 0 # No Of Testing Data

# Read The Csv File and Create Training Data and Testing Data
with open("givenData.csv") as file:
    reader = csv.reader(file) # Read The CSV
    reader = list(reader)
    reader = reader[1:] # remove the Header
    nTestingData = math.floor(len(reader)*0.25) # Divide the no of Data
    for row in reader:
        t = {'text':'','label':0} # Temporary Dictonary
        text = bs4.BeautifulSoup(row[0],'html.parser').text  # Convert the HTML into Text
        t['text'] = text.split() # Divide The Text Message into Words
        if row[1] == 'Spam':  # if it is Spam change the label to 1
            t['label'] = 1
        trainingData.append(t) # Add the temporary dictonary into the Trainimg Data
    file.close() # close the CSV File
random.shuffle(trainingData) # Shuffle the Data

testingData = trainingData[:nTestingData] # Create a Testing Dataset
trainingData = trainingData[nTestingData:] # Create a Training Dataset

calssifier.Train(trainingData) # Train The Classifier
# calssifier.readData();

# Testing the Classifier
total = len(testingData) # get The total length of data in testing Data 
correct = 0 # Variable to store no of correct Guesses

for data in testingData: # itterate for each data in Testing dataset
    x = calssifier.Predict(data['text']) # Predict the Value
    if x == True:# increase the value of correct if predection is correct
        if data['label'] == 1:
                correct += 1 
    else:
        if data['label'] == 0:
                correct += 1
err = correct/total # calculate the error
# Display analytics
print("\tAnalytics on Testing Data\t")
print("ACCURACY :  "+str(err*100)+" %")
print("Error    :  "+str( (1-err)*100)+" %")

calssifier.saveData()