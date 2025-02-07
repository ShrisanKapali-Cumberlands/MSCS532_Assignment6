# MSCS532 Algorithms and Data Structures
# Assignment 6 - Medians and Order Statistics & Elementary Data
# Shrisan Kapali
# Student Id: 005032249

# Importing random & time library
import random
import time
import sys
import matplotlib.pyplot as plt
import numpy as np


# Deterministic algorith Median of Medians for finding kth smallest in worst-case linear time
def median_of_medians(array, k):
    # Dividing the array into 5 sublists
    sublists = [array[i : i + 5] for i in range(0, len(array), 5)]
    medians = [sorted(sublist)[len(sublist) // 2] for sublist in sublists]

    if len(medians) <= 5:
        pivot = sorted(medians)[len(medians) // 2]
    else:
        pivot = median_of_medians(medians, len(medians) // 2)

    # Partitioning left and right
    lows = [x for x in array if x < pivot]
    highs = [x for x in array if x > pivot]

    if k < len(lows):
        return median_of_medians(lows, k)
    elif k > len(lows):
        return median_of_medians(highs, k - len(lows) - 1)
    else:
        return pivot
