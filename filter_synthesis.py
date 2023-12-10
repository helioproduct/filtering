import math
import numpy as np

def normalize_index(index, kernel_size):
    return (2 * index - kernel_size + 1) / kernel_size

def new_filter(kernel_size, filter_func, *args):
    kernel = []

    for x in range(kernel_size):
        kernel_row = [0] * kernel_size

        for y in range(kernel_size):
            normalized_x = normalize_index(x, kernel_size)
            normalized_y = normalize_index(y, kernel_size)

            if args:
                kernel_row[y] = filter_func(normalized_x, normalized_y, *args)
            else:
                kernel_row[y] = filter_func(normalized_x, normalized_y)
        kernel.append(kernel_row)

    return np.array(kernel)

def normalize_filter(kernel):
    norma = np.sum(kernel)
    return kernel / norma