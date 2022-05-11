from bs4 import BeautifulSoup
import requests
import csv
from random import randint
from time import sleep

dictionary = {}
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

    # map ids to respective view counts
    for i in range(len(id_list)):
        url = "https://www.youtube.com/watch?v=" + id_list[i]
        views = scrape(url)
        sleep(1)
        if views == 0:
            break
        dictionary[id_list[i]] = views
        print(i, views)


if __name__ == "__main__":
    main()