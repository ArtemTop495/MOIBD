from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
import json
import pymupdf
import io
import re
import glob

all_json = glob.glob(r"Data\*.json")

data = ['' for i in range(len(all_json))]
for i in range(0, len(all_json)-1):
    with open(f'{all_json[i]}', 'r', encoding="utf8") as file:
        data[i] = json.load(file)

def parse_json(local_data):
    for i in local_data['refs']:
        all_ref_text.append(i[0])
        all_ref_day.append(i[1]['day'])
        all_ref_month.append(i[1]['month'])
        all_ref_time.append(i[1]['time'])
        all_ref_rate.append(i[2]['rate'])
        all_ref_views.append(i[2]['views'])

all_ref_text = []
all_ref_day = []
all_ref_month = []
all_ref_time = []
all_ref_rate = []
all_ref_views = []
for i in range(0, len(data)-1):
    parse_json(data[i])

df = pd.DataFrame({
    'ref text': all_ref_text,
    'ref day': all_ref_day,
    'ref month': all_ref_month,
    'ref time': all_ref_time,
    'ref rate': all_ref_rate,
    'ref views': all_ref_views
})
df

df.to_csv("articles_and_games.csv", index=False)
df2 = pd.read_csv("articles_and_games.csv")
df2.head()