import requests
from bs4 import BeautifulSoup
import pymysql
import re
from datetime import datetime
import math
import random
import configparser
config = configparser.ConfigParser()
config.read("../newsnow_config.ini")

ip = config['Connect']['server']
port = config['Connect'].getint('port')
db = config['Connect']['db']
user = config['User']['username']
password  = config['User']['password']

# 資料庫參數設定
db_settings = {
    "host": ip,
    "port": port,
    "user": user,
    "password": password,
    "db": db,
    "charset": "utf8"
}

news_link = []
page_number = 2
count = 0

conn = pymysql.connect(**db_settings)
sql = conn.cursor()

HTML_PARSER = "html.parser"
session_requests = requests.session()

def one_day_loop():
    global unix_now_time
    global target_url
    # 因為只隔一天，會有30分鐘的落差沒有爬取到，所以抓1個小時(3600)，每天多爬一個小時，確保不會遺漏資料：86400+3600
    unix_now_time -= 28800
    print("###---------------------------------------------------------------------------------------------------------------------------###")
    print("日期：" + str(datetime.fromtimestamp(unix_now_time).strftime("%Y-%m-%d-%HH-%MM-%SS")) + "，開始爬取！")
    print("###---------------------------------------------------------------------------------------------------------------------------###")

    target_url = ROOT_URL + '&d=' + str(unix_now_time)
    print("目標網址：" + target_url)
    return

def get_news_times():
    global time_list
    global news_time
    global timestamp
    import time as tik
    global unix_now_time
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'cookie' : 'nn_uid=ID=20230113124108:20031; uky=c_101-52-51-53-55-52-53-55; __gads=ID=a8f508913da9aae2:T=1673613768:S=ALNI_MaXhnKshaWNI3_B4S2tsL7ZrjUHyQ; _gid=GA1.3.1866699498.1674844904; cookie_policy=2; nnusrst=c_Pubs._QhZ-ArtList.1164452301;1164383991-CookieConsent.2; nn_sid=05f39165b0957402573c; __gpi=UID=00000ba35459f50e:T=1673613768:RT=1675190027:S=ALNI_MYgrQ3BB6jRaIPrz_4wPOGfjbi8mQ; nn_ssn=c_Qh; _ga=GA1.3.1553473696.1673583694; _ga_ZS7YYS67BF=GS1.1.1675180683.14.1.1675193984.0.0.0; NNNewsfeedHistory=c__C.326-A.Cyber+Security-K.Qu-S.1675194287-N.15653-Y.UKZ-_C.1-A.Technology-K.Qu-S.1675168527-N.15630-Y.UK; NN_Eng=0'
    }
    list_req = requests.get(target_url, headers=headers)
    if list_req.status_code == requests.codes.ok:
        soup = BeautifulSoup(list_req.content, HTML_PARSER)
    
    time_list =[]
    times = soup.find_all('span', 'time',limit=amount)
    for time in times:
        time_list.append(time['data-time'])
    print(time_list)
    print("###-----------------------------------------------------------------------------------------------------------------------------------------------###")
    print("爬取自：" + str(datetime.fromtimestamp(unix_now_time+ 3600)) + " 到 " + str(datetime.fromtimestamp(unix_now_time - 28800)) + "的資料")
    print("###-----------------------------------------------------------------------------------------------------------------------------------------------###")

    for i in range(count,len(time_list)):
        delay_choices_1 = [10, 20, 12, 11]
        delay_1 = random.choice(delay_choices_1)
        print("休息時間：" + str(delay_1))
        tik.sleep(delay_1)
        news_time = int(time_list[i])
        
        if news_time <= (unix_now_time+3600) and news_time > (unix_now_time - 28800):
            timestamp = str(datetime.fromtimestamp(news_time))
            print("Local-Date-time : " + str(timestamp))
            print("UNIX-Timestamp : " + str(news_time))
            get_news_link(i)
        else:
            print("###-----------------------------------------------------------------------------------------------------------------------------------------------###")
            print("已爬取完畢！")
            print("###-----------------------------------------------------------------------------------------------------------------------------------------------###")
            break
            

def get_news_link(j):
    global url
    global title
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'cookie' : 'nn_uid=ID=20230113124108:20031; uky=c_101-52-51-53-55-52-53-55; __gads=ID=a8f508913da9aae2:T=1673613768:S=ALNI_MaXhnKshaWNI3_B4S2tsL7ZrjUHyQ; _gid=GA1.3.1866699498.1674844904; cookie_policy=2; nnusrst=c_Pubs._QhZ-ArtList.1164452301;1164383991-CookieConsent.2; nn_sid=05f39165b0957402573c; __gpi=UID=00000ba35459f50e:T=1673613768:RT=1675190027:S=ALNI_MYgrQ3BB6jRaIPrz_4wPOGfjbi8mQ; nn_ssn=c_Qh; _ga=GA1.3.1553473696.1673583694; _ga_ZS7YYS67BF=GS1.1.1675180683.14.1.1675193984.0.0.0; NNNewsfeedHistory=c__C.326-A.Cyber+Security-K.Qu-S.1675194287-N.15653-Y.UKZ-_C.1-A.Technology-K.Qu-S.1675168527-N.15630-Y.UK; NN_Eng=0'
    }
    list_req = requests.get(target_url, headers=headers)
    if list_req.status_code == requests.codes.ok:
        soup = BeautifulSoup(list_req.content, HTML_PARSER)

    divs = soup.find_all(href =re.compile("^https://c.newsnow.co.uk/A/"),limit=amount)
    url = divs[j].get('href')
    title = divs[j].text.replace('\n', '')
    title = re.sub(r'[^\x00-\x7f]', r' ', title)
    print("Title : " + title)
    print("URL : " + url)

    get_news_contents()
    print('-------------------------------------------------------')

def get_news_contents():
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'cookie' : 'nn_uid=ID=20230113124108:20031; uky=c_101-52-51-53-55-52-53-55; __gads=ID=a8f508913da9aae2:T=1673613768:S=ALNI_MaXhnKshaWNI3_B4S2tsL7ZrjUHyQ; _gid=GA1.3.1866699498.1674844904; cookie_policy=2; nnusrst=c_Pubs._QhZ-ArtList.1164452301;1164383991-CookieConsent.2; nn_sid=05f39165b0957402573c; __gpi=UID=00000ba35459f50e:T=1673613768:RT=1675190027:S=ALNI_MYgrQ3BB6jRaIPrz_4wPOGfjbi8mQ; nn_ssn=c_Qh; _ga=GA1.3.1553473696.1673583694; _ga_ZS7YYS67BF=GS1.1.1675180683.14.1.1675193984.0.0.0; NNNewsfeedHistory=c__C.326-A.Cyber+Security-K.Qu-S.1675194287-N.15653-Y.UKZ-_C.1-A.Technology-K.Qu-S.1675168527-N.15630-Y.UK; NN_Eng=0'
    }

    content =''
    session_requests = requests.session()
    #使用GET方法取得網頁資訊
    response = session_requests.get(url,headers=headers)
    print("Response status is : " + str(response.status_code))

    if response.status_code == requests.codes.ok:
        soup = BeautifulSoup(response.content,"html.parser")
    # 取得response中所有的超連結，存在 a_tags。並 pop 出第一個網址
    a_tags = soup.find_all('a')
    a_tags.reverse()
    web = a_tags.pop().get('href')
    print("trans_url is : " + web)

    news_url = requests.get(web, headers=headers)
    if news_url.status_code == requests.codes.ok:
        soup = BeautifulSoup(news_url.content,"html.parser")

    source = soup.find_all('p')
    for x in source:
         content += x.text + '\n'
    content = re.sub(r'[^\x00-\x7f]', r' ', content)
    content = (content[:18000]) if len(content) > 18000 else content
    Top = 0
    Popular = 0

    try:
        insert = \
                "INSERT INTO newsnow_security (title,url,trans_url,response_status,post_date,timestamp,content,TOP,POPULAR) VALUES (%s, %s, %s, %s, %s, %s, %s,%s,%s)"
        data = (title, url,web,response.status_code,timestamp , news_time, content,Top,Popular)
        sql.execute(insert, data)
        conn.commit()
    except pymysql.IntegrityError as ex:
        error_message = str(ex)
        if 'Duplicate entry' in error_message:
            print("重複資料! 繼續執行")
        else:
            raise  # 如果不是重複資料，則繼續執行
    except Exception as ex:
        print(ex)
    finally:
        content = ''

# 取得參數，實際執行指令：python ./news_now_crawler_2.py -d [爬幾天]
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--days", required = True,
                help = "How many days do you want to search?",
                type = int)
parser.add_argument("-t", "--time", required = True,
                help = "From now on or a specific time?",
                type = int)
args = parser.parse_args()
days = args.days
# 時間，將其轉換為 unix timestamp，給他加一小時，讓他多爬
unix_now_time = args.time

# 目標連線網址：type=ln 表示爬取 latest 的新聞
ROOT_URL = 'https://www.newsnow.co.uk/h/Technology/Cyber+Security?type=ln'
target_url = ROOT_URL + '&d=' + str(unix_now_time)
amount = math.inf

print(f"爬蟲爬取次數(一次一天)：至今前{ days:^3} 天，每天爬取 {amount} 篇新聞(若數量為inf，則代表每天爬取當次新聞) ")
print("目標網址：" + target_url)

days = days*3

for x in range(days):
    print("目前時間：" + str(datetime.fromtimestamp(unix_now_time)) + " / UNIX時間：" + str(unix_now_time))
    print('-------------------------------------------------------')
    get_news_times()
    # 更改時間
    one_day_loop()

sql.close()
conn.close()

