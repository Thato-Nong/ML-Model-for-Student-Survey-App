import pandas as pd
from faker import Faker
import random

# Initialize Faker for generating random text
fake = Faker()

# Generate synthetic data with 100 responses
responses = [fake.paragraph(nb_sentences=3, variable_nb_sentences=True, ext_word_list=None) for _ in range(100)]
happiness_labels = [random.choice([0, 1]) for _ in range(100)]

# Create a DataFrame
df = pd.DataFrame({'response': responses, 'happiness': happiness_labels})

# Save the DataFrame to a CSV file
df.to_csv('large_student_responses.csv', index=False)

print("CSV file with 100 responses generated successfully.")
