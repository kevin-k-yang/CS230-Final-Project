# Authors: Aryan Chaudhary (achaud@stanford.edu)
#          Kevin K Yang    (kevvyang@stanford.edu)
#          Brian Wu        (brian.wu@stanford.edu)
#
# Description: This file includes the code that we use to sort our datasets. Initially we had the value MILLI = 1000000 to create an
# up-down 1,000,000 for our binary classifier, but then we repurposed this to be a log scale multiple classifier
#     five:    10K < x < 100K
#     six:     100K < x < 1M
#     seven:   1M < x < 10M
#     eight:   10M < x < 100M

import os
import csv
import numpy as np

SOURCE = "images"
DESTINATION_5 = "images/five/"
DESTINATION_6 = "images/six/"
DESTINATION_7 = "images/seven/"
DESTINATION_8 = "images/eight/"

FILENAME = "viewcounts.csv"
MILLI = 1000000

def parse_view_counts(link):
    view_count_map = {}
    with open(link) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        first_line = True
        for row in csv_reader:
            if not first_line:
                log_value = int(np.log10(int(row[1])))
                view_count_map[row[0]] = log_value
            else:
                first_line = False
    return view_count_map

# separate images based on virality (>= million views)
def main():
    viewcounts_map = parse_view_counts(FILENAME)

    image_files = os.listdir(SOURCE)
    for image_file in image_files:
        files = os.listdir(SOURCE + "/" + image_file)
        for f in files:
            image_name = f[:f.find(".")]
            if viewcounts_map[image_name] == 5:
                os.rename(SOURCE + "/" + image_file + "/" + f, DESTINATION_5 + f)
            elif viewcounts_map[image_name] == 6:
                os.rename(SOURCE + "/" + image_file + "/" + f, DESTINATION_6 + f)
            elif viewcounts_map[image_name] == 7:
                os.rename(SOURCE + "/" + image_file + "/" + f, DESTINATION_7 + f)
            elif viewcounts_map[image_name] == 8:
                os.rename(SOURCE + "/" + image_file + "/" + f, DESTINATION_8 + f)


if __name__ == "__main__":
    main()