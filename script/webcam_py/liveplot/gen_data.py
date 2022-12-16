import csv
import random
import time

x_value = 0
total_1 = 1000
total_2 = 1000

field_names = ['x_value', 'total_1', 'total_2']

with open('data.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
    csv_writer.writeheader()
while True:
    with open('data.csv', 'a') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
        info = {
            'x_value': x_value,
            'total_1': total_1,
            'total_2': total_2
        }
        csv_writer.writerow(info)
        print(x_value, total_1, total_2)
        x_value += 1
        total_1 += random.randint(-10, 10)
        total_2 += random.randint(-10, 10)
    time.sleep(0.2)