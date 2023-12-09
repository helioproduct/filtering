import numpy as np

def linear_blur(x, y):
    radius = np.sqrt(x**2 + y**2)
    if radius > 1:
        return 0
    return 1 - radius


''' kernel_size >= 6 * Sigma + 1'''
def gauss_blur(x, y, sigma):
    return np.exp(-(x**2 + y**2)/(2*sigma**2))

def hyperbolic_blur(x, y, k):
    radius = np.sqrt(x**2 + y**2)
    if radius > 1:
        return 0
    return (x-1)**2*(k*x+k+1)/(k+1)/(k*x+1)

''' kernel_size >= 5 * sigma'''
def log_sharpen(x, y, sigma):
    return (x**2 + y**2 - 2 * sigma**2) / sigma**4 * np.exp(-(x**2 + y**2)/(2*sigma**2))
