import numpy as np

# Define how to get the phase space for a pixel

# Define how to draw a random direction inside a pixel

# Define Monte Carlo integration for a pixel

# Loop over pixels
    # Draw n random rays per pixel. For each one
        # Save the ray directions
        # Compute and save the acceptance

# Loop over all rays and construct the matrix G in ray space


# Forward modelling is then:
    # Apply G to the model
    # get intensity for all rays
    # For each pixel:
        # Sum up intensity * acceptance over rays in that pixel. Divide by n cause Monte Carlo integration.
        # be done with a matrix multiplication for all pixels at once.
        # or a loop with numba.


