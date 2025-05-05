from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

# Настройка headless-режима
options = Options()
options.add_extension("C:\\Users\\artembruh\\Downloads\\Augmented Steam - Chrome Web Store 4.2.1.0.crx")
options.headless = True
driver = webdriver.Chrome(options=options)

# Открытие страницы Steam
url = 'https://store.steampowered.com/search/?category1=998%2C994&filter=topsellers&ndl=1'
driver.get(url)
time.sleep(10)

# Прокрутка страницы вниз, чтобы загрузить 100 игр
SCROLL_PAUSE_TIME = 2
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(SCROLL_PAUSE_TIME)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height or len(driver.find_elements("class name", "search_result_row")) >= 100:
        break
    last_height = new_height

# Получение HTML и парсинг
soup = BeautifulSoup(driver.page_source, 'html.parser')
driver.quit()

games = soup.find_all('a', class_='search_result_row')[:100]

# Парсим каждую игру
for idx, game in enumerate(games, 1):
    title = game.find('span', class_='title').text.strip()
    release_date = game.find('div', class_='search_released').text.strip()
    app_url = game['href']

    # Парсим страницу самой игры для описания и тегов
    try:
        driver = webdriver.Chrome(options=options)
        driver.get(app_url)
        time.sleep(2)
        game_soup = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit()

        description_tag = game_soup.find("div", {"id": "game_area_description"})
        description = description_tag.text.replace('\n', ' ').replace('\t','').replace('About This Game','').replace('About This Software','').strip() if description_tag else 'No description'

        tags = [tag.text.strip() for tag in game_soup.select('.glance_tags.popular_tags a')]
        review_text = game_soup.find('meta', itemprop='ratingValue').attrs['content'] + ' / 10'

    except Exception as e:
        description = "N/A"
        tags = []
        review_text = "N/A"

    print(f"{idx}. {title}")
    print(f"   Дата выхода: {release_date}")
    print(f"   Описание: {description}")
    print(f"   Теги: {', '.join(tags)}")
    print(f"   Оценка: {review_text}")
    print("-" * 80)