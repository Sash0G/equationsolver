import numpy as np

# Example matrix
matrix = np.array([[1, 2, 3, 4, 5],
                   [6, 7, 8, 9, 10],
                   [11, 12, 13, 14, 15],
                   [16, 17, 18, 19, 20],
                   [21, 22, 23, 24, 25]])

# Define the size of the square region
size = 3

# Extract the square regions from each corner
top_left_square = matrix[:size, :size]
top_right_square = matrix[:size, -size:]
bottom_left_square = matrix[-size:, :size]
bottom_right_square = matrix[-size:, -size:]

# Create a new matrix from the corner squares
new_matrix = np.block([[top_left_square, top_right_square],
                       [bottom_left_square, bottom_right_square]])

print(new_matrix)