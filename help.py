import csv

data = [
    ["Male", "Computer Science", 4, 3, 5, 4, 8],
    ["Female", "Psychology", 3, 4, 4, 3, 6],
    ["Male", "Engineering", 5, 2, 3, 5, 7],
    ["Female", "Art", 2, 5, 4, 2, 5],
    ["Male", "Physics", 4, 3, 3, 4, 7],
    ["Female", "Chemistry", 3, 4, 5, 3, 6],
]

header = ["gender", "major", "academic_experience", "extracurricular_activities", "social_life", "overall_well_being", "happiness"]

# Write to CSV file
with open('student_survey_data_generated.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Write header
    csvwriter.writerow(header)
    
    # Write data
    csvwriter.writerows(data)

print("CSV file generated successfully.")
