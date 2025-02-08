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


# Test cases of different array sizes and distributions
# Setting up the sizes
sizes = [
    100000,
    200000,
    300000,
    400000,
    500000,
    600000,
    700000,
    800000,
]
distributions = ["Sorted", "Reverse_Sorted", "Random"]

# Creating a list to store the execution time for each sort
executionTimeDeterministic = {dist: [] for dist in distributions}
executionTimeRandomized = {dist: [] for dist in distributions}


# Looping through all the sizes
for size in sizes:
    # For each distribution generate the data sets
    for dist in distributions:
        if "Sorted" == dist:
            data = list(range(size))
        elif "Reverse_Sorted" == dist:
            data = list(range(size, 0, -1))
        elif "Random" == dist:
            data = random.sample(range(size), size)

        # Finding the kth element
        k = 50

        # Perform Deterministic quick sort
        start = time.time()
        print(f"The {k}th element is ", median_of_medians(data.copy(), k))
        end = time.time()
        executionTimeDeterministic[dist].append(end - start)
        print(
            f"Execution time of deterministic median-of-median sort for finding {k} th element in "
            + dist
            + " array of size "
            + str(size)
            + " took "
            + str(end - start)
            + " seconds"
        )

        # Perform Randomized quick select
        start = time.time()
        print(
            f"The {k}th element is ",
            randomized_select(data.copy(), 0, len(data) - 1, k),
        )
        end = time.time()
        executionTimeRandomized[dist].append(end - start)
        print(
            f"Execution time of randomized quickselect sort for finding {k} th element in "
            + dist
            + " array of size "
            + str(size)
            + " took "
            + str(end - start)
            + " seconds"
        )
        print("")


# Using matplot library to plot the graph of the execution time
plt.figure(figsize=(10, 6))
for dist in distributions:
    plt.plot(
        sizes,
        executionTimeDeterministic[dist],
        label=f"Deterministic Median-Of-Median - {dist}",
        linestyle="--",
    )
    plt.plot(
        sizes,
        executionTimeRandomized[dist],
        label=f"Randomized QuickSelect - {dist}",
        linestyle="-",
    )

plt.xlabel("Input Size")
plt.ylabel("Time (seconds)")
plt.title("Deterministic vs Randomized Selection sort to fing 50th element Performance")
plt.legend()
plt.grid()
plt.show()


## Part2 - Implementing array, stacks, queues, and linked lists
## Implementing arrays
class Array:
    # Initialize array of fixed size with values as None
    def __init__(self, size):
        self.size = size
        self.array = [None] * size

    # Insert element at array at index
    def insert(self, index, value):
        if 0 <= index < self.size:
            self.array[index] = value
        else:
            raise IndexError("Index out of bounds")

    # Delete element from the index, resetting its value to None
    def delete(self, index):
        if 0 <= index < self.size:
            self.array[index] = None
        else:
            raise IndexError("Index out of bounds")

    # Get the element at index
    def access(self, index):
        if 0 <= index < self.size:
            return self.array[index]
        else:
            raise IndexError("Index out of bounds")


## Test cases for array
print("\n\n************************************")
print("Test cases for Array")
print("Initializing an empty array of size 10")
start = time.time()
array = Array(10)
end = time.time()
executionTime = end - start

print(
    "Array initialize of size 10. Current array size = ",
    array.size,
)
print("Execution time to initialize in seconds = ", executionTime)
print("Current value of index 0 : Expected : None | Actual :", array.access(0))

# Adding elements into array, and accessing their value
array.insert(0, "First element")
array.insert(1, "Second element")
print("Current value of index 0 : Expected : First element | Actual :", array.access(0))
print(
    "Current value of index 1 : Expected : Second element | Actual :", array.access(1)
)

# Removing element 0
array.delete(0)
print(
    "Current value of index 0 after deletion: Expected : None | Actual :",
    array.access(0),
)


# Implementing stacks from scratch
class Stack:
    # Initialize an empty stack
    def __init__(self):
        self.stack = []

    # Push an element
    def push(self, value):
        self.stack.append(value)

    # Pop the last element
    def pop(self):
        # Check if the stack is not empty
        if not len(self.stack) == 0:
            return self.stack.pop()
        else:
            raise IndexError("Stack is empty")

    # Peek the last element
    def peek(self):
        if not len(self.stack) == 0:
            return self.stack[-1]
        else:
            return None


## Test cases for stack
print("\n\n************************************")
print("Test cases for Stack")
print("Initializing an empty stack ")
stack = Stack()
# Adding elements to stack
stack.push("First element")
stack.push("Second element")
stack.push("Third element")
# Popping elements from stack
print("Stack peek", stack.peek())
print("Stack pop", stack.pop())
print("Stack pop", stack.pop())
print("Stack peek", stack.peek())


# Implementing Queues
class Queue:
    # Initializing an empty queue
    def __init__(self):
        self.queue = []

    # Add element to a queue
    def enqueue(self, value):
        self.queue.append(value)

    # Remove the first element from queue
    def dequeue(self):
        if not len(self.queue) == 0:
            return self.queue.pop(0)
        else:
            return IndexError("Empty queue")

    # Peek the element at index 0
    def peek(self):
        if not len(self.queue) == 0:
            return self.queue[0]
        else:
            return None


## Test cases for Queue
print("\n\n************************************")
print("Test cases for Queue")
print("Initializing an empty queue")
queue = Queue()
# Adding elements to a queue
queue.enqueue("First element")
queue.enqueue("Second element")
queue.enqueue("Third element")
queue.enqueue("Fourth element")
# Queue peek
print("Queue peek", queue.peek())
# Dequeue
print("Queue dequeue", queue.dequeue())
print("Queue dequeue", queue.dequeue())
print("Queue dequeue", queue.dequeue())
print("Queue peek", queue.peek())


# Implementing Linked List
# Implementing Node
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def insert(self, value):
        new_node = Node(value)
        if self.head is None:
            self.head = new_node
            return
        else:
            new_node.next = self.head
            self.head = new_node

    def delete(self, value):
        temp = self.head
        if temp and temp.value == value:
            self.head = temp.next
            temp = None
            return

        prev = None
        while temp and temp.value != value:
            prev = temp
            temp = temp.next

        if temp is None:
            return

        prev.next = temp.next
        temp = None

    def traverse(self):
        temp = self.head
        while temp:
            print(temp.value)
            temp = temp.next


## Test cases for Linked List
print("\n\n************************************")
print("Test cases for Linked List")
print("Initializing an empty Linked List")
linkedList = LinkedList()
print("Empty linked list. Expected: None | Actual: ", linkedList.traverse())
# Adding value to linked list
linkedList.insert(1)
linkedList.insert(2)
linkedList.insert(3)
linkedList.insert(4)
print("Adding 4 entries. Calling the traverse function")
print(linkedList.traverse())

print("Removing 2. The new list is")
linkedList.delete(2)
print(linkedList.traverse())


# Rooted Trees
class TreeNode:
    # Initialize a tree with value and childrens
    def __init__(self, value):
        self.value = value
        self.children = []

    # Add new child
    def add_child(self, child):
        self.children.append(child)

    # Traverse
    def traverse(self, level=0):
        print(" " * level + str(self.value))
        for child in self.children:
            child.traverse(level + 1)


## Test cases for Tree
print("\n\n************************************")
print("Test cases for Tree")
print("Initializing an empty tree")
# Initializing root tree
root = TreeNode("Root")
# Create child roots
child1 = TreeNode("Child1")
child2 = TreeNode("Child2")
child3 = TreeNode("Child3")
# Add children to root tree
root.add_child(child1)
root.add_child(child2)
root.add_child(child3)
# Add sub child to child trees
child1.add_child(TreeNode("Child1.1"))
child1.add_child(TreeNode("Child1.2"))
child2.add_child(TreeNode("Child2.1"))
child2.add_child(TreeNode("Child2.2"))
child3.add_child(TreeNode("Child3.1"))
child3.add_child(TreeNode("Child3.2"))

print("Tree Structure:")
root.traverse()
