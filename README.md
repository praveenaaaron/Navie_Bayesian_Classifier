
## 📌 OVERVIEW

This program:

* Loads a CSV file with numerical data.
* Splits it into training and test sets.
* Calculates probabilities for each class (Naive Bayes).
* Predicts class labels on unseen data.
* Computes accuracy.

---

## 🧠 NAIVE BAYES THEORY (brief)

Naive Bayes assumes:

* Each feature contributes independently to the outcome (the "naive" part).
* For numeric features, it uses **Gaussian distribution**:

  $$
  P(x|\mu, \sigma) = \frac{1}{\sqrt{2\pi}\sigma} \cdot e^{-\frac{(x - \mu)^2}{2\sigma^2}}
  $$

---

## 🧩 CODE BREAKDOWN

### 1. **Load CSV**

```python
def loadcsv(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset
```

* Loads rows from the file.
* Converts all values to floats (numbers).

---

### 2. **Split into Training and Test Set**

```python
def splitdataset(dataset, splitratio):
    trainsize = int(len(dataset) * splitratio)
    trainset = []
    copy = list(dataset)
    while len(trainset) < trainsize:
        index = random.randrange(len(copy))
        trainset.append(copy.pop(index))
    return [trainset, copy]
```

* Randomly splits data into training and testing using `splitratio` (e.g. 0.67 = 67% train, 33% test).

---

### 3. **Group Data by Class**

```python
def separatebyclass(dataset):
    separated = {}
    for vector in dataset:
        class_value = vector[-1]
        if class_value not in separated:
            separated[class_value] = []
        separated[class_value].append(vector)
    return separated
```

* Groups rows by their class label (e.g. 0 or 1).
* Each group is used to calculate class-specific statistics.

---

### 4. **Summarize Dataset**

```python
def mean(numbers):
    return sum(numbers) / len(numbers)

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / (len(numbers) - 1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries = [(mean(col), stdev(col)) for col in zip(*dataset)]
    del summaries[-1]  # remove class column summary
    return summaries
```

* Computes **mean and standard deviation** of each feature (column).
* Skips the class label.

---

### 5. **Summarize By Class**

```python
def summarizebyclass(dataset):
    separated = separatebyclass(dataset)
    summaries = {}
    for classvalue, instances in separated.items():
        summaries[classvalue] = summarize(instances)
    return summaries
```

* Creates a `summary` (mean, stdev) per feature **per class**.

---

### 6. **Probability Functions**

```python
def calculateprobability(x, mean, stdev):
    if stdev == 0:
        return 1.0 if x == mean else 1e-10  # avoid division by zero
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
```

* Computes **Gaussian probability** of a value `x` given `mean` and `stdev`.

---

### 7. **Class Probability Calculation**

```python
def calculateclassprobabilities(summaries, inputvector):
    probabilities = {}
    for classvalue, classsummaries in summaries.items():
        probabilities[classvalue] = 1
        for i in range(len(classsummaries)):
            mean, stdev = classsummaries[i]
            x = inputvector[i]
            probabilities[classvalue] *= calculateprobability(x, mean, stdev)
    return probabilities
```

* For each class, calculates the combined probability of the input vector.
* Multiplies all feature probabilities.

---

### 8. **Prediction**

```python
def predict(summaries, inputvector):
    probabilities = calculateclassprobabilities(summaries, inputvector)
    return max(probabilities, key=probabilities.get)
```

* Picks the class with the highest total probability.

---

### 9. **Make Predictions and Evaluate**

```python
def getpredictions(summaries, testset):
    return [predict(summaries, row) for row in testset]

def getaccuracy(testset, predictions):
    correct = sum([1 for i in range(len(testset)) if testset[i][-1] == predictions[i]])
    return (correct / len(testset)) * 100.0
```

* Predicts for each test row.
* Compares predictions to actual class values to get accuracy.

---

### 10. **Main Driver Function**

```python
def main():
    filename = 'naivedata.csv'
    splitratio = 0.67
    dataset = loadcsv(filename)
    trainingset, testset = splitdataset(dataset, splitratio)
    summaries = summarizebyclass(trainingset)
    predictions = getpredictions(summaries, testset)
    accuracy = getaccuracy(testset, predictions)
    print('Accuracy of the classifier is : {0}%'.format(accuracy))
```

* Loads data
* Trains the model
* Predicts outcomes
* Prints accuracy

---

## ✅ OUTPUT EXAMPLE

```
Split 10 rows into train=6 and test=4 rows  
Accuracy of the classifier is : 75.0%
```


