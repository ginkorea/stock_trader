import numpy as np
from sympy import divisors

# Helper function to find the least common divisor and pad if necessary
def find_best_head_size(input_size):
    # Try common divisors starting from an ideal 64
    for divisor in range(64, 1, -1):
        if input_size % divisor == 0:
            return divisor
    # If no suitable divisor is found, default to 1 head (not ideal but valid)
    return 1

# Function to pad the input if the size is not ideal
def pad_input(x, input_size):
    num_samples, time_steps, _ = x.shape
    padding_size = (64 - (input_size % 64)) if input_size % 64 != 0 else 0
    if padding_size > 0:
        new_input_size = input_size + padding_size
        X_padded = np.pad(x, ((0, 0), (0, 0), (0, padding_size)), 'constant')
        return X_padded, new_input_size
    return x, input_size

# Verify the output labels before training
def verify_labels(y):
    y_list = y.tolist() if isinstance(y, np.ndarray) else y
    y_flat = np.array(y_list).flatten()
    unique_labels = set(y_flat)
    unique_labels_list = list(unique_labels)
    print(f"Unique labels head: {unique_labels_list[:5]}{'...' if len(unique_labels_list) > 5 else ''}")
    if len(unique_labels) <= 1:
        raise ValueError("There is an issue with the target labels: Only one or zero unique values found.")
    else:
        print(f"Labels verification passed with {len(unique_labels)} unique values.")
