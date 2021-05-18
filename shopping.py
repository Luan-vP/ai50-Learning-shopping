import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    evidence = []
    labels = []

    with open(filename) as file:
        csv_reader = csv.reader(file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            
            if line_count > 0:

                (new_evidence, new_label) = process_row(row)
                evidence.append(new_evidence)
                labels.append(new_label)

            line_count += 1 

    return (evidence, labels)


def process_row(row):
    """
    Given a row of data from a csv file, return a processed data row and label as a tuple.
    """
    processed_row = []
    months = ["Jan","Feb","Mar","Apr","May","June","Jul","Aug","Sep","Oct","Nov","Dec"]

    # Int and float columns can be directly converted
    # Other columns are parsed manually
    # Revenue Column becomes the 'label'

    for index in range(len(row)):

        if index in [0,2,4,11,12,13,14]:
            processed_row.append(int(row[index]))
        elif index in [1,3,5,6,7,8,9]:
            processed_row.append(float(row[index]))
        elif index == 10:
            # Month
            processed_row.append(months.index(row[index]))
        elif index == 15:
            # VisitorType
            if row[index] == "Returning_Visitor":
                processed_row.append(1)
            else:
                processed_row.append(0)
        elif index == 16:
            # Weekend
            if row[index] == "TRUE":
                processed_row.append(1)
            else:
                processed_row.append(0)
        elif index == 17:
            # Revenue
            if row[index] == "TRUE":
                label = 1
            else: 
                label = 0
        else:
            raise "Incorrect data array length"

    return (processed_row, label)
            


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """

    neigh = KNeighborsClassifier(n_neighbors = 1)

    neigh.fit(evidence, labels)

    return neigh


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    difference = [ label - prediction for label, prediction in zip(predictions, labels) ]

    print(difference)

    # Sensitivity is true positive rate
    # number of correctly predicted 1s

    positive_labels = labels.count(1)

    incorrect_positive_predictions = difference.count(1)

    sensitivity = ( positive_labels - incorrect_positive_predictions ) / positive_labels

    # Specificity is true negative rate
    # number of correctly proedicted 0s

    negative_labels = labels.count(0)

    incorrect_negative_predictions = difference.count(-1)

    specificity = ( negative_labels - incorrect_negative_predictions ) / negative_labels

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
