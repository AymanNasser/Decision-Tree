import pandas as pd
import numpy as np
from datetime import datetime
import time

class Node:

    def __init__(self, data):

        self.left = None
        self.right = None
        self.data = data


    def PrintTree(self,level = 0):
        strlevel = "  "*level+"⤷"
        if(self!=None):
            if not isinstance(self,Node):
                print(strlevel,self)
            else:
                print(strlevel,self.data)
                level+=1
                Node.PrintTree(self.left,level)
                Node.PrintTree(self.right,level)
           
            
            

# Preparing data
dataFrame = pd.read_csv("sample_train.csv")
dataFrame = dataFrame.drop("reviews.text", axis=1)
data = dataFrame.values

# Return the major element (eg. most repeated one) in data
def classifyData(data):
    labelPerColumn = data[:, -1]
    uniqueResult, countUniqueResult = np.unique(labelPerColumn, return_counts=True)
    
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

def calcEntropy(data):
    labelPerColumn = data[:, -1]
    _, counts = np.unique(labelPerColumn, return_counts=True)
    probabilities = counts / counts.sum()
    Entropy = sum(probabilities * -np.log2(probabilities))

    return Entropy


def calcOverallEntropy(equal_Data, notEqual_Data):

    n = len(equal_Data) + len(notEqual_Data)
    p_data_equal = len(equal_Data) / n
    p_data_notEqual = len(notEqual_Data) / n

    overall_entropy = (p_data_equal * calcEntropy(equal_Data) + p_data_notEqual * calcEntropy(notEqual_Data))

    return overall_entropy


def determineOptimumSplit(data, uniqueSplits):
    overallEntropy = 1000
    for columnIndex in range(len(uniqueSplits)):
        for value in uniqueSplits[columnIndex]:
            #value= uniqueSplits[columnIndex][i]
            equal_Data, notEqual_Data = splitData(data, splitColumn= columnIndex, splitValue=value)
            currentOverallEntropy = calcOverallEntropy(equal_Data, notEqual_Data)

            if currentOverallEntropy <= overallEntropy:
                overallEntropy = currentOverallEntropy
                bestSplit_Column = columnIndex
                bestSplit_Value = value

    return bestSplit_Column, bestSplit_Value


#print(determineOptimumSplit(data,retrieveUniqueSplits(data)))


# this function run the algorithm and builds the tree recursively
def decisionTree(dataFrame, counter=0, minSamples=5, maxDepth=5):
    # data preparations
    if counter == 0:
        # build list of labels
        global COLUMN_HEADERS
        COLUMN_HEADERS = dataFrame.columns
        # convert pandas data frame to numpy 2d array
        
        data = dataFrame.values
    else:
        # df is already converted to 2darray
        data = dataFrame

        # base cases
    if (len(data) < minSamples) or (counter == maxDepth):
        classification = classifyData(data)
        return classification

    else:
        counter += 1

        uniqueSplits = retrieveUniqueSplits(data)
        split_column, split_value = determineOptimumSplit(data, uniqueSplits)
        dataEqual, dataNotEqual = splitData(data, split_column, split_value)

        if len(dataEqual) == 0 or len(dataNotEqual) == 0:
            classification = classifyData(data)
            return classification

        # instantiate new node
        question = "{} = {}".format(COLUMN_HEADERS[split_column], split_value)

        sub_tree = Node(question)

        # find answers (recursion)
        yes_answer = decisionTree(dataEqual, counter, minSamples, maxDepth)
        no_answer = decisionTree(dataNotEqual, counter, minSamples,maxDepth)

        # If the answers are the same, then there is no point in asking the question.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base cases).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree.left =(yes_answer)
            sub_tree.right = (no_answer)

        return sub_tree


#pprint(decisionTree(dataFrame))


def classify_example(example, tree):

    # base case
    if not isinstance(tree,Node):
        return tree

    question = tree.data
    feature_name, _, value = question.split(" ")

    if str(example[feature_name]) == str(value):
       
        answer = tree.left
    else:
        answer = tree.right

    # recursive part
    residual_tree = answer
    return classify_example(example, residual_tree)

# using the sample_dev.csv 
def calcAccuracy(tree):
    
    dataFrame = pd.read_csv("sample_dev.csv")
    dataFrame = dataFrame.drop("reviews.text", axis=1)
    dataFrame["prediction"] = dataFrame.apply(classify_example, axis=1, args=(tree,))
    dataFrame["prediction-correct"] = dataFrame["prediction"] == dataFrame["rating"]
    accuracy = dataFrame["prediction-correct"].mean()
    return accuracy


#using sample_test.csv
def generatePredictionFile(tree):
        dataFrame = pd.read_csv("sample_test.csv")
        dataFrame = dataFrame.drop("reviews.text", axis=1)
        np.savetxt(r'test_prediction',dataFrame.apply(classify_example, axis=1, args=(tree,)),fmt='%s')

def prepareInput(dataFrame):
    values = []

    values.append(input().split())
# #to enter feature value one by one
#     values.append([])
#     for feature in dataFrame.columns:
#         if(feature != "rating"):
#             values[0].append(input("{}: ".format(feature)))
#         else:  values[0].append("-1")
    
    df = pd.DataFrame(values,columns =dataFrame.columns)
   
    df = df.drop(df.columns[-1], axis=1)
    
    return df

def runProject():
    timeBeforeCalc = datetime.now().time()
    start_time = time.time()
    timeBeforeCalc = timeBeforeCalc.strftime("%H:%M:%S")
    print("Starting Time ", timeBeforeCalc)
    tree = decisionTree(dataFrame, maxDepth=7)
    accuracy = calcAccuracy(tree)
    timeAfterCalc = datetime.now()
    timeAfterCalc = timeAfterCalc.strftime("%H:%M:%S")
    print("Ending Time ", timeAfterCalc)
    print("Training Time = {:.2f} seconds".format(time.time() - start_time))
    print("Accuracy: {:.2f}%".format(100 * accuracy))
    print("Decision Tree")
    tree.PrintTree()
    generatePredictionFile(tree)

    a = input("do you want to enter review? \n No = 0 \n Yes = 1 \n")
    if a == "1":
        print("enter only 0 or 1")
        df = prepareInput(dataFrame)
        print(classify_example(df, tree))

runProject()
