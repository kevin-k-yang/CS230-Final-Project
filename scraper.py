from bs4 import BeautifulSoup
import requests
import csv
from random import randint
from time import sleep

# input url and output num views of video
def scrape(url):
  soup = BeautifulSoup(requests.get(url).text, "html.parser")
  findViews = soup.find("meta", itemprop="interactionCount")
  if findViews is None:
    return 0
  return findViews['content']

def main():
    # read ids from metadata file
    filename = 'metadata.csv'
    id_list = []
    with open(filename, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            id_list.append(row[0])

    id_list = id_list[1:]

    count = 0
    with open('viewcounts.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            count += 1
    

    with open('viewcounts2.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        for i in range(count + 1, len(id_list)):
            url = "https://www.youtube.com/watch?v=" + id_list[i]
            views = scrape(url)
            sleep(randint(1, 5))
            if views == 0:
                break
            print(i, views)
            writer.writerow([id_list[i], views])


if __name__ == "__main__":
    main()