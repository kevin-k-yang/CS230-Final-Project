import os
import csv

SOURCE = "images"
DESTINATION_0 = "images/Below/"
DESTINATION_1 = "images/Above/"

FILENAME = "viewcounts.csv"
MILLI = 1000000

def parse_view_counts(link):
    view_count_map = {}
    with open(link) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        first_line = True
        for row in csv_reader:
            if not first_line:
                view_count_map[row[0]] = 1 if int(row[1]) >= MILLI else 0
            else:
                first_line = False
    return view_count_map

def main():
    viewcounts_map = parse_view_counts(FILENAME)

    image_files = os.listdir(SOURCE)
    for image_file in image_files:
        files = os.listdir(SOURCE + "/" + image_file)
        for f in files:
            image_name = f[:f.find(".")]
            if viewcounts_map[image_name] == 0:
                os.rename(SOURCE + "/" + image_file + "/" + f, DESTINATION_0 + f)
            else:
                os.rename(SOURCE + "/" + image_file + "/" + f, DESTINATION_1 + f)

if __name__ == "__main__":
    main()