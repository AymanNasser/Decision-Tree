import pandas as pd
import numpy as np

# Preparing data
dataFrame = pd.read_csv("sample_train.csv")
dataFrame = dataFrame.drop("reviews.text", axis=1)


data = dataFrame.values

# Return the major element (eg. most repeated one) in data
def classifyData(data):

    labelPerColumn = data[:, -1]

    uniqueResult, countUniqueResult = np.unique(labelPerColumn, return_counts=True)
    #print(countUniqueResult[1] / (countUniqueResult[0] + countUniqueResult[1]))

    indexMostRepeated_Result = countUniqueResult.argmax()
    return uniqueResult[indexMostRepeated_Result]


def retrieveUniqueSplits(data):

    
    rowsNumber, colsNumber = data.shape
    uniqueSplits = [None]*(colsNumber-1) # *Not Dictionary*
    for colIndex in range(colsNumber - 1):
        values = data[:, colIndex]
        # Removing duplicates of each feature
        unique_values = np.unique(values)

        uniqueSplits[colIndex] = unique_values

    return uniqueSplits


def splitData(data, splitColumn, splitValue):

    splitColumn_values = data[:, splitColumn]
    equal_Data = data[splitColumn_values == splitValue]
    notEqual_Data = data[splitColumn_values != splitValue]

    return equal_Data, notEqual_Data


