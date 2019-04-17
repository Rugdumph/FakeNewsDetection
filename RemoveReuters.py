import csv
import re


with open("News_dataset/TrueClean.csv", mode='w') as write_file:
	writer = csv.writer(write_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	with open("News_dataset/True.csv") as read_file:
			csv_reader = csv.reader(read_file, delimiter=',')
			for row in csv_reader:
				writer.writerow([row[0], re.sub(r'\w*\s*\(Reuters\) - ',"",row[1],count=1), row[2], row[3]])
