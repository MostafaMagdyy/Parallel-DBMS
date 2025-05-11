import os
import csv
import random
import datetime
from faker import Faker
import string

# Initialize Faker for generating realistic data
fake = Faker()

# Function to create a directory if it doesn't exist
def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

# Function to generate a random date between start and end dates
def random_date(start_date, end_date):
    time_between = end_date - start_date
    days_between = time_between.days
    random_days = random.randrange(days_between)
    return start_date + datetime.timedelta(days=random_days)

# Function to generate data for a table and save it as CSV
def generate_table_data(table_name, columns, num_rows, directory):
    # Define file path
    file_path = os.path.join(directory, f"{table_name}.csv")
    
    # Open file and write data
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header with type indicators and primary key format
        header = []
        for col in columns:
            if col.get('is_primary', False):
                header.append(f"{col['name']} ({col['type']}) (P)")
            else:
                header.append(f"{col['name']} ({col['type']})")
        writer.writerow(header)
        
        # Generate data rows
        for i in range(num_rows):
            row = []
            for col in columns:
                col_type = col['type']
                if col_type == 'N':
                    # Numerical: Generate random salary or ID-like number
                    if 'salary' in col['name'].lower():
                        value = round(random.uniform(30000, 120000), 2)
                    else:
                        value = random.randint(1, 10000) if col.get('is_primary', False) else random.randint(0, 1000)
                elif col_type == 'D':
                    # Date: Generate random date between 2020 and 2023
                    start_date = datetime.datetime(2020, 1, 1)
                    end_date = datetime.datetime(2023, 12, 31)
                    value = random_date(start_date, end_date).strftime('%Y-%m-%d')
                elif col_type == 'T':
                    # Text: Generate name, department, or random string
                    if 'name' in col['name'].lower():
                        value = fake.name()
                    elif 'department' in col['name'].lower():
                        value = random.choice(['HR', 'Engineering', 'Sales', 'Marketing', 'Finance'])
                    else:
                        value = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
                row.append(value)
            writer.writerow(row)
    
    print(f"Generated CSV file: {file_path} with {num_rows} rows")

# Main function to generate data for 3 tables
def main():
    # Define directory for CSV files
    directory = "./csv_data"
    create_directory(directory)
    
    # Define table structures
    tables = [
        # {
        #     'name': 'employees',
        #     'columns': [
        #         {'name': 'id', 'type': 'N', 'is_primary': True},
        #         {'name': 'name', 'type': 'T'},
        #         {'name': 'salary', 'type': 'N'},
        #         {'name': 'hire_date', 'type': 'D'},
        #         {'name': 'department_id', 'type': 'N'}
        #     ],
        #     'num_rows': 1000
        # },
        # {
        #     'name': 'departments',
        #     'columns': [
        #         {'name': 'id', 'type': 'N', 'is_primary': True},
        #         {'name': 'name', 'type': 'T'}
        #     ],
        #     'num_rows': 5
        # },
        {
            'name': 'projects',
            'columns': [
                {'name': 'id', 'type': 'N', 'is_primary': True},
                {'name': 'project_name', 'type': 'T'},
                {'name': 'start_date', 'type': 'D'},
                {'name': 'budget', 'type': 'N'}
            ],
            'num_rows': 4
        }
    ]
    
    # Generate data for each table
    for table in tables:
        generate_table_data(table['name'], table['columns'], table['num_rows'], directory)

if __name__ == "__main__":
    # Call the main function to generate data
    main()
    
    # Calculate and print salary statistics
    employee_file = "./csv_data/employees.csv"
    if os.path.exists(employee_file):
        min_salary = float('inf')
        max_salary = float('-inf')
        sum_salary = 0
        count = 0
        
        with open(employee_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                salary = float(row[2])  # Salary is in the third column (index 2)
                min_salary = min(min_salary, salary)
                max_salary = max(max_salary, salary)
                sum_salary += salary
                count += 1
        
        print(f"\nSalary Statistics:")
        print(f"Min Salary: ${min_salary:.2f}")
        print(f"Max Salary: ${max_salary:.2f}")
        print(f"Sum of all Salaries: ${sum_salary:.2f}")
        print(f"Average Salary: ${sum_salary/count:.2f}")