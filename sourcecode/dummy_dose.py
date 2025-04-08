import numpy as np

def generate_dummy_dose(shape=(64, 64, 64)):
    # Generate a dummy dose distribution with random values.
    dose = np.random.rand(*shape)
    return dose
