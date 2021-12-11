import random
import csv

file = "TSLA.csv"
with open(file, newline='') as f:
    reader = csv.reader(f)
    teslaData = list(reader)

MIN_OPEN = 3.23
MAX_OPEN = 891

MIN_HIGH = 3.33
MAX_HIGH = 900

MIN_LOW = 3
MAX_LOW = 872

MIN_CLOSE = 3.16
MAX_CLOSE = 883

MIN_ADJ_CLOSE = 3.16
MAX_ADJ_CLOSE = 883

MIN_VOLUME = 593000
MAX_VOLUME = 305000000

trainData = []
testData = []

k = 8
epochs = 1000

teslaData.pop(0)
random.shuffle(teslaData)

def changeRange(input, minInput, maxInput, minOutput, maxOutput):
    output = minOutput + ((maxOutput - minOutput) /
                          (maxInput - minInput)) * (input - minInput)
    return output

for i in range(len(teslaData)):
    if i!=0:
        current = []
        # open
        current.append(changeRange(float(teslaData[i][1]), MIN_OPEN, MAX_OPEN, 0, 1))
        # high (thats what we're predicting)
        high = changeRange(float(teslaData[i][2]), MIN_HIGH, MAX_HIGH, 0, 1)
        # low
        current.append(changeRange(float(teslaData[i][3]), MIN_LOW, MAX_LOW, 0, 1))
        # close
        current.append(changeRange(float(teslaData[i][4]), MIN_CLOSE, MAX_CLOSE, 0, 1))
        # adjusted close
        current.append(changeRange(float(teslaData[i][5]), MIN_ADJ_CLOSE, MAX_ADJ_CLOSE, 0, 1))
        # volume
        current.append(changeRange(float(teslaData[i][6]), MIN_VOLUME, MAX_VOLUME, 0, 1))
        # add the answer to the end of the list
        current.append(high)
        if i % k == 0:
            testData.append(current)
        else:
            trainData.append(current)

class RegressionModel:
    def __init__(self, numberOfInputs):
        self.weights = []
        self.bias = 0
        self.learning_rate = 0.1
        for _ in range(numberOfInputs):
            self.weights.append(random.uniform(0, 1))

    def printWeights(self):
        for i in self.weights:
            print(i)

    # data is a list of numbers of lists, each of which has length numberOfInputs+1, because output is the last one
    def train(self, dataPoints):
        for data in dataPoints:
            guess = 0
            if (len(data)-1!=len(self.weights)):
                print("INPUTS AND WEIGHTS ARE NOT THE SAME LENGTH")
                print(len(data)-1, len(self.weights))
                return

            for i in range(len(self.weights)):
                guess += self.weights[i]*data[i]
            guess += self.bias

            answerIndex = len(data)-1
            error = (data[answerIndex] - guess)
            for i in range(len(self.weights)):
                self.weights[i] += error * data[i] * self.learning_rate
            self.bias += error * self.learning_rate
            
    # exclude answer from inputs here
    def test(self, input):
        sum = 0
        for i in range(len(self.weights)):
            sum += self.weights[i]*input[i]
        sum += self.bias
        sum -= input[len(input)-1]
        return abs(sum)

# Pick a random sample from the test data
randomExample = testData[random.randint(0, len(testData))]

# Create the model
model = RegressionModel(5)

print("Loss before training:", model.test(randomExample))

# train model over number of epochs
for _ in range(epochs):
    model.train(trainData)

print("Loss after training:", model.test(randomExample))








