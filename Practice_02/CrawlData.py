# from bs4 import BeautifulSoup
# import requests
# import urllib.request
#
#
# keyword = 'sleeping'
# path = 'DataSet/'
# user_agent = 'Mozilla/5.0 (Linux; U; Android 4.0.2; en-us; Galaxy Nexus Build/ICL53F) AppleWebKit/534.30 (KHTML, like Gecko) Version/4.0 Mobile Safari/534.30'
# headers = {'User-Agent': user_agent}
#
# source = requests.get('https://www.freeimages.com/search/' + keyword, headers=headers).text
# soup = BeautifulSoup(source, 'lxml')
#
# Images = []
# img_links = soup.select('img[src^="https://www.freeimages.com/image"]')
#
# for i in range(len(img_links)):
#     Images.append(img_links[i]['src'])
#
# for i in range(len(Images)):
#     if i > 100: break
#     name = path + str(i) + '.jpg'
#     urllib.request.urlretrieve(Images[i], name)



from icrawler.builtin import GoogleImageCrawler
from bs4 import BeautifulSoup
import os

def download_images(keyword, save_dir, num_images=200):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    google_crawler = GoogleImageCrawler(parser_threads=10, downloader_threads=20, storage={'root_dir': save_dir})

    google_crawler.crawl(keyword=keyword, max_num=num_images)

def rename_images(save_dir, keyword):
    image_files = [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))]

    for i, filename in enumerate(image_files):
        _, file_extension = os.path.splitext(filename)
        new_filename = f"{keyword}{i+1:05d}{file_extension}"
        os.rename(os.path.join(save_dir, filename), os.path.join(save_dir, new_filename))

if __name__ == "__main__":
    keyword = "sleeping"
    save_directory = "image/" + keyword

    download_images(keyword, save_directory, num_images=300)
    rename_images(save_directory, keyword)