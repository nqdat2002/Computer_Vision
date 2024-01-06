import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os

def download_image(url, save_folder):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        filename = url.split("/")[-1]
        filepath = os.path.join(save_folder, filename)

        with open(filepath, 'wb') as file:
            for chunk in response.iter_content(chunk_size=128):
                file.write(chunk)
        print(f"Đã tải xuống: {filename}")
    else:
        print(f"Lỗi khi tải xuống ảnh từ: {url}")

def download_images(keyword, num_images=100):
    base_url = "https://storage.googleapis.com/openimages/web/visualizer/index.html?type=detection&set=train&c=%2Fm%2F05r655"
    response = requests.get(base_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        img_tags = soup.find_all('img')

        save_folder = keyword.replace(" ", "_")
        os.makedirs(save_folder, exist_ok=True)

        count = 0
        for img_tag in img_tags:
            img_url = urljoin(base_url, img_tag['src'])
            download_image(img_url, save_folder)
            count += 1
            if count >= num_images:
                break

    else:
        print(f"Lỗi khi truy cập trang web: {base_url}")

download_images('Girl', num_images=100)
