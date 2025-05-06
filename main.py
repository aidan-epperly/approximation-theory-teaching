import numpy as np
import math
from functools import partial

PI = math.pi
TWO_PI = 2 * PI
ONE_OVER_TWO_PI = 1 / (2 * PI)

'''
I know this is basically trivial, but it's worth
including anyway.
'''
def cyclic_trapezoid_integrator(function, grid):
    total_sum = 0 + 0j
    for point in grid:
        total_sum += function(point)
    return total_sum * (grid[1] - grid[0])

def fourier_projection(function, max_degree, integrator):
    grid = np.linspace(0, TWO_PI, 2 * max_degree, False)
    coefficient_array = (0 + 0j) * np.zeros(2 * max_degree + 1)
    
    for k in range(2 * max_degree + 1):
        coefficient_array[k] = ONE_OVER_TWO_PI * integrator(lambda point :
                                          function(point) * np.exp(1j * -(k - max_degree) * point),
                                          grid)

    return coefficient_array

def fourier_projection_function(coefficient_array, point):
    total_sum = 0
    degree = (len(coefficient_array) - 1) / 2
    for index in range(len(coefficient_array)):
        total_sum += coefficient_array[index] * np.exp(1j * (index - degree) * point)
    return total_sum

def main():
    degree = 5
    function = lambda point : math.sin(point)
    coefficient_array = fourier_projection(function, degree, cyclic_trapezoid_integrator)
    grid = np.linspace(0, TWO_PI, 2 * degree + 1)
    projection = lambda point : fourier_projection_function(coefficient_array, point)
    print("Coefficients", coefficient_array, "\n")
    print("Grid Points", grid, "\n")
    print("True Function Values", np.vectorize(function)(grid), "\n")
    print("Approximate Function Values", np.real(np.vectorize(projection)(grid)), "\n")

if __name__ == "__main__":
    main()
