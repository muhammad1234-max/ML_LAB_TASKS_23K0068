import pandas as pd
import numpy as np

# Method 1: Using NumPy structured array 
data = np.array([
    (35, 4, 5, 60, 2),
    (40, 5, 6, 70, 3),
    (25, 3, 4, 50, 1),
    (30, 4, 3, 40, 1),
    (20, 2, 2, 30, 0),
    (28, 3, 4, 45, 2),
    (38, 4, 5, 65, 2),
    (22, 2, 3, 35, 1),
    (33, 4, 4, 55, 2),
    (27, 3, 2, 40, 1)
], dtype=[('test_score', 'i4'), 
          ('writing_skills', 'i4'), 
          ('reading_skills', 'i4'),
          ('attendance', 'i4'), 
          ('study_hours', 'i4')])

df1 = pd.DataFrame(data)
print("DataFrame created using NumPy structured array")
print(df1)


#Method 2: Using pandas Series of dicts 
records = [
    {'test_score': 35, 'writing_skills': 4, 'reading_skills': 5, 'attendance': 60, 'study_hours': 2},
    {'test_score': 40, 'writing_skills': 5, 'reading_skills': 6, 'attendance': 70, 'study_hours': 3},
    {'test_score': 25, 'writing_skills': 3, 'reading_skills': 4, 'attendance': 50, 'study_hours': 1},
    {'test_score': 30, 'writing_skills': 4, 'reading_skills': 3, 'attendance': 40, 'study_hours': 1},
    {'test_score': 20, 'writing_skills': 2, 'reading_skills': 2, 'attendance': 30, 'study_hours': 0},
    {'test_score': 28, 'writing_skills': 3, 'reading_skills': 4, 'attendance': 45, 'study_hours': 2},
    {'test_score': 38, 'writing_skills': 4, 'reading_skills': 5, 'attendance': 65, 'study_hours': 2},
    {'test_score': 22, 'writing_skills': 2, 'reading_skills': 3, 'attendance': 35, 'study_hours': 1},
    {'test_score': 33, 'writing_skills': 4, 'reading_skills': 4, 'attendance': 55, 'study_hours': 2},
    {'test_score': 27, 'writing_skills': 3, 'reading_skills': 2, 'attendance': 40, 'study_hours': 1}
]

df2 = pd.DataFrame(pd.Series(records))
print("\n DataFrame created using Series of dicts ")
print(df2)


#Method 3: Using from_records()
tuple_data = [
    (35, 4, 5, 60, 2),
    (40, 5, 6, 70, 3),
    (25, 3, 4, 50, 1),
    (30, 4, 3, 40, 1),
    (20, 2, 2, 30, 0),
    (28, 3, 4, 45, 2),
    (38, 4, 5, 65, 2),
    (22, 2, 3, 35, 1),
    (33, 4, 4, 55, 2),
    (27, 3, 2, 40, 1)
]

df3 = pd.DataFrame.from_records(tuple_data, columns=['test_score','writing_skills','reading_skills','attendance','study_hours'])
print("\nDataFrame created using from_records() ")
print(df3)
