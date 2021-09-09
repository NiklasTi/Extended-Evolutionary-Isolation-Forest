import numpy as np
import time
import scipy.io
import csv

from EEIF import EEIF


# Calculates the precision and recall based on the results for a threshold
def get_precision_and_recall(results, labels):
    true_positive_count = 0
    true_negative_count = 0
    false_positive_count = 0
    false_negative_count = 0

    for idx in range(len(labels)):

        prediction = results[idx]
        label = labels[idx]

        if label == prediction:
            if label == 1:
                true_positive_count += 1
            else:
                true_negative_count += 1
        else:
            if label == 1:
                false_negative_count += 1
            else:
                false_positive_count += 1

    if (true_positive_count == 0):
        return 0, 0, true_positive_count, true_negative_count, false_negative_count, false_positive_count
    else:
        precision = true_positive_count / (true_positive_count + false_positive_count)
        recall = true_positive_count / (true_positive_count + false_negative_count)

        return precision, recall, true_positive_count, true_negative_count, false_negative_count, false_positive_count


# Writes the anomaly scores to a csv file
def score_list(scores, labels):
    results_string = "../results/EEIF_scores.csv"
    auc_list = np.stack((s, lab), axis=-1)
    with open(results_string, 'a', newline='') as csvfile:
        for i in range(len(auc_list)):
            results_csv = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            results_csv.writerow(auc_list[i])
        csvfile.close()


# The function to run the EEIF code. The dataset, iterations and thresholds can be adjusted here

def run():
    data_set = scipy.io.loadmat('.mat')                                                  # Adjust the dataset as needed
    data = np.array(data_set['X'])
    labels = data_set['y']
    iteration = 100                                                                      # Adjust Iterations as needed
    threshold = 0.25                                                                     # Adjust Threshold as needed

    current_time = time.time()

    AD = EEIF(
        data,
        labels,
        iteration,
        threshold
    )

    results, scores = AD.run_full_test(data)

    score_list(scores, labels)
    precision, recall, true_positive_count, true_negative_count, false_negative_count, false_positive_count = get_precision_and_recall(results, labels)

    return precision, recall, true_positive_count, true_negative_count, false_negative_count, false_positive_count


# Executes the run() function and writes the resulting precision and recall as well as TP, TN, FN and FP into a csv file
p, r, tp, tn, fn, fp = run()
total_results = p, r, tp, tn, fn, fp

results_string = '../results/EEIF_precision_recall.csv'
f = open(results_string, "x")
f.close()
headers = ['Precision', 'Recall', 'TP', 'TN', 'FN', 'FP']
with open(results_string, 'a', newline='') as csvfile:
    results_csv = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    results_csv.writerow(headers)
    results_csv.writerow(total_results)
    csvfile.close()
