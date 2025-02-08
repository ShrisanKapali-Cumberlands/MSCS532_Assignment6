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

    # Partitioning
    lows = [x for x in array if x < pivot]
    highs = [x for x in array if x > pivot]

    # If the kth smallest is in lows
    if k < len(lows):
        return median_of_medians(lows, k)

    # If the kth smallest is in higs
    elif k > len(lows):
        return median_of_medians(highs, k - len(lows) - 1)

    # If the pivot is the kth smallest
    else:
        return pivot


# Randomized Quickselect
def partition(array, low, high):
    # Select the high as the pivot
    pivot = array[high]
    i = low - 1

    for j in range(low, high):
        # if array at index j is less than pivot, increment
        if array[j] <= pivot:
            i += 1
            array[i], array[j] = array[j], array[i]

    array[i + 1], array[high] = array[high], array[i + 1]
    return i + 1


def randomized_partition(array, low, high):
    # select a random pivot between low and high and swap with las element
    pivot_index = random.randint(low, high)
    array[pivot_index], array[high] = array[high], array[pivot_index]
    return partition(array, low, high)


# Finally a function for randomized quickselect to find kth element
def randomized_select(array, low, high, k):
    if low == high:
        return array[low]

    pivot_index = randomized_partition(array, low, high)

    # the kth element is at pivot
    if k == pivot_index:
        return array[k]
    elif k < pivot_index:
        return randomized_select(array, low, pivot_index - 1, k)
    else:
        return randomized_select(array, pivot_index + 1, high, k)
