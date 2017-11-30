import csv
import os


for filename in os.listdir("."):
    if filename.endswith(".txt"): 
        f2 = open(filename, "w")
        writer = csv.writer(f2, delimiter=';')

        first = True
        with open("train/"+filename, "r") as csv_file:
            reader = csv.reader(csv_file, delimiter=';')
            for row in reader:
                if first:
                    writer.writerow(row)
                    first = False
                else:
                    date = row[1].split('-')
                    year = int(date[0])
                    month = int(date[1])
                    day = int(date[2].split(' ')[0])
                    if(month <= 9 and day<24):
                        writer.writerow(row)

