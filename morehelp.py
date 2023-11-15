import pandas as pd
import numpy as np

# Your existing dataset
data = pd.read_csv('large_student_responses.csv')

# Separate the dataset into happy and unhappy responses
happy_responses = data[data['happiness'] == 1]
unhappy_responses = data[data['happiness'] == 0]

# Decide on the number of responses to keep for each class (adjust as needed)
num_responses_to_keep = min(len(happy_responses), len(unhappy_responses))

# Randomly sample from each class to balance the dataset
balanced_data = pd.concat([
    happy_responses.sample(n=num_responses_to_keep, random_state=22),
    unhappy_responses.sample(n=num_responses_to_keep, random_state=22)
])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced dataset to a new CSV file
balanced_data.to_csv('input.csv', index=False)
