from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
import json
import pymupdf
import io
import re
import glob
# GET - запрос
url = 'https://store.steampowered.com/search/?l=russian?filter=topsellers&ndl=1'
page = requests.get(url)
page.encoding = 'utf-8'
soup = bs(page.text, 'html.parser')
result_list = {'title': [], 'description': [], 'date': [], 'tags': [], 'rating': []}
games = soup.find_all('a', class_='search_result_row')[:200]
for idx, game in enumerate(games, 1):
    title = game.find('span', class_='title').text.strip()
    if title.__contains__('Steam Deck'):
        continue
    result_list['title'].append(title)
    date = game.find('div', class_='search_released').text.strip()
    result_list['date'].append(date)
    game_url = game['href']
    game_page = requests.get(game_url[:game_url.find('?')] + '?l=russian')
    game_soup = bs(game_page.text, 'html.parser')
    description_tag = game_soup.find("div", {"id": "game_area_description"})
    description = description_tag.text.replace('\n', ' ').replace('\t','').replace('About This Game','').replace('About This Software','').strip() if description_tag else 'No description'
    result_list['description'].append(description)
    tags = ', '.join([tag.text.strip() for tag in game_soup.select('.glance_tags.popular_tags a')])
    result_list['tags'].append(tags)
    if game_soup.find('div', class_='summary column').text.__contains__('No'):
        review_text = 'No user reviews'
    else:
        review_text = game_soup.find('meta', itemprop='ratingValue').attrs['content'] + ' / 10'
    result_list['rating'].append(review_text)
file_name = 'games.csv'
df = pd.DataFrame(data=result_list)
df.to_csv(file_name)
df.head()
# загрузка в list пути всех нужных pdf файлов
all_pdf = glob.glob(r"PDF2\*.pdf")
pattern = r'(\d{1,2})\s([а-я]{3})\sв\s(\d{2}:\d{2})'
len(all_pdf)
for i in all_pdf:
    pdf_document = i
    doc = pymupdf.open(pdf_document)
    for current_page in range(len(doc)):
        page = doc.load_page(current_page)
        page_text = page.get_text("text")

def extract_text_from_pdf(pdf):
    doc = pymupdf.open(pdf)
    text = ''
    for current_page in range(len(doc)):
        page = doc.load_page(current_page)
        page_text = page.get_text("text")
        text += page_text
    return text.replace('8 минTAU15', '').replace('+1','1').replace('+9','9').replace('9 мин','').replace('+7','7').replace('Средний1','').replace('Простой','').replace('1.1K','1100').replace('3 мин','').replace('Простой  3 мин','').replace('Cloud4Y OSINT','OSINT').replace('Cloud4Y 5 часов назад','5 часов назад').replace('КАК СТАТЬ АВТОРОМ','').replace('1 мин','').replace('Простой9 мин','').replace('8 мин','').replace('5 мин','').replace('zubarek','').replace('Средний18 мин','').replace('5 минArnak', '').replace('redmadrobot','').replace('ru_vds','').replace('Средний19 минartyomsoft', '').replace('Средний18 минredmadrobot', '').replace('5 минzubarek', '').replace('technokratiya','').replace('5 часов назад\tCloud4Y','5 часов назад').replace('Простой3 мин','').replace('Exosphere ','').replace('AlfaTeam ','').replace('AnnieBronson ','').replace('redmadrobot ','').replace('red_mad_robot ','').replace('Средний18 мин ','').replace('artyomsoft','').replace('Arnak','').replace('RUVDS.com','').replace('VDS/VPS-хостинг. Скидка 15% по коду HABR15','').replace('Средний19 мин','').replace('RUVDS.com VDS/VPS-хостинг. Скидка 15% по коду HABR15','').replace('TAU15','').replace('RUVDS.com VDS/VPS-хостинг. Скидка 15% по коду HABR15','')


def pars_pdf(text):
    text2 = text.split('\n')
    text2 = [x for x in text2 if x.strip()]
    name = ''
    if 'Рейтинг' in text2:
        num = text2.index('Рейтинг')
        raiting = text2[num - 1]
    for i in range(0, len(text2) - 1):
        if 'назад' in text2[i] and name == '':
            num3 = i
            date_publish = text2[num3].strip()
            if text2[num3 + 1].__contains__('Cloud4Y'):
                name = text2[num3 + 2] + ' ' + text2[num3 + 3]
            else:
                if not f'{text2[num3 + 2]}'.strip().isalnum():
                    name = text2[num3 + 1] + ' ' + text2[num3 + 2]
                else:
                    name = text2[num3 + 1]
        elif re.search(pattern, text2[i]):
            num3 = i
            date_publish = text2[num3].strip()
            if text2[num3 + 1].__contains__('Cloud4Y'):
                name = text2[num3 + 2] + ' ' + text2[num3 + 3]
            else:
                if not f'{text2[num3 + 2]}'.strip().isalnum():
                    name = text2[num3 + 1] + ' ' + text2[num3 + 2]
                else:
                    name = text2[num3 + 1]
        if 'Хабы' in text2[i]:
            hubs = ''
            j = i
            while not f'{text2[j]}'.strip().isnumeric() and not text2[j].__contains__('Редакторский дайджест'):
                hubs += text2[j] + ' '
                j += 1
            break
    return name, raiting, date_publish, hubs

Brak = []
result_list1 = {'title': [], 'description': [], 'date': [], 'tags': [], 'rating': []}

for pdf in all_pdf:
    try:
        x = extract_text_from_pdf(pdf)
        N, R, D, H = pars_pdf(x)
        D2 = ''.join(x).replace('\n', ' ')
        result_list1['title'].append(N)
        result_list1['rating'].append(R)
        result_list1['date'].append(D)
        result_list1['description'].append(D2)
        result_list1['tags'].append(H)
        result_list['title'].append(N)
        result_list['rating'].append(R)
        result_list['date'].append(D)
        result_list['description'].append(D2)
        result_list['tags'].append(H)
    except Exception as e:
        print(e)
        Brak.append(pdf)

file_name = 'pdf_files.csv'
df = pd.DataFrame(data=result_list1)
df.to_csv(file_name)
file_name = 'pdf_and_games.csv'
df1 = pd.DataFrame(data=result_list)
df1.to_csv(file_name)

all_json = glob.glob(r"Data\*.json")

from types import NoneType
def parse_json(local_data, file_name):
    for i in local_data['refs']:
        all_ref_text.append(i[0])
        all_ref_day.append(i[1]['day'])
        all_ref_month.append(i[1]['month'])
        all_ref_time.append(i[1]['time'])
        all_ref_rate.append(i[2]['rate'])
        all_ref_views.append(i[2]['views'])
        result_list['title'].append(file_name)
        result_list['rating'].append(i[2]['rate'])
        result_list['date'].append(f'{i[1]['day']} {i[1]['month']} {i[1]['time']}')
        result_list['description'].append(i[0])
        if type(local_data['info']) is NoneType:
            result_list['tags'].append('')
        else:
            result_list['tags'].append(local_data['info']['industries'])

data = ['' for i in range(len(all_json))]
all_ref_text = []
all_ref_day = []
all_ref_month = []
all_ref_time = []
all_ref_rate = []
all_ref_views = []
for i in range(0, len(all_json)-1):
    with open(f'{all_json[i]}', 'r', encoding="utf8") as file:
        data[i] = json.load(file)
for i in range(0, len(data)-1):
    parse_json(data[i], all_json[i])

df = pd.DataFrame({
    'ref text': all_ref_text,
    'ref day': all_ref_day,
    'ref month': all_ref_month,
    'ref time': all_ref_time,
    'ref rate': all_ref_rate,
    'ref views': all_ref_views
})

df.to_csv("articles_and_games.csv", index=False)
df2 = pd.read_csv("articles_and_games.csv")
df2.head()

file_name = 'pdf_and_games_and_json.csv'
df2 = pd.DataFrame(data=result_list)
df2.to_csv(file_name)