import requests
from bs4 import BeautifulSoup
import pymysql
import base64
import re
from datetime import datetime
import time
import math
import random
import time
import os

HTML_PARSER = "html.parser"

ROOT_URL = 'https://www.newsnow.co.uk/h/Technology/Cyber+Security?type=ln'

news_link = []

page_number = 2

passwd = ''



conn = pymysql.connect(host='localhost', user='root', passwd = '', db='ids_cti')
sql = conn.cursor()


count = 0
session_requests = requests.session()
execute_time = datetime.now()
trans_time = int(round(execute_time.timestamp()))
target = ROOT_URL + '&d=' + str(trans_time)

top_news = "https://www.newsnow.co.uk/h/Technology/Cyber+Security"

def one_month_loop():
    global target
    global trans_time
    trans_time -= 86400
    target = ROOT_URL + '&d=' + str(trans_time)
    #print(target)
    #print(trans_time)

def get_news_times():
    global time_list
    global news_time
    global timestamp
    import time as tik
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'cookie' : 'nn_uid=ID=20230113124108:20031; uky=c_101-52-51-53-55-52-53-55; __gads=ID=a8f508913da9aae2:T=1673613768:S=ALNI_MaXhnKshaWNI3_B4S2tsL7ZrjUHyQ; _gid=GA1.3.1866699498.1674844904; cookie_policy=2; nnusrst=c_Pubs._QhZ-ArtList.1164452301;1164383991-CookieConsent.2; nn_sid=05f39165b0957402573c; __gpi=UID=00000ba35459f50e:T=1673613768:RT=1675190027:S=ALNI_MYgrQ3BB6jRaIPrz_4wPOGfjbi8mQ; nn_ssn=c_Qh; _ga=GA1.3.1553473696.1673583694; _ga_ZS7YYS67BF=GS1.1.1675180683.14.1.1675193984.0.0.0; NNNewsfeedHistory=c__C.326-A.Cyber+Security-K.Qu-S.1675194287-N.15653-Y.UKZ-_C.1-A.Technology-K.Qu-S.1675168527-N.15630-Y.UK; NN_Eng=0'
    }
    list_req = requests.get(target, headers=headers)
    if list_req.status_code == requests.codes.ok:
        soup = BeautifulSoup(list_req.content, HTML_PARSER)
    
    time_list =[]
    times = soup.find_all('span', 'time',limit=amount)
    for time in times:
        time_list.append(time['data-time'])
    
    for i in range(count,len(time_list)):
        delay_choices_1 = [5, 10, 20, 12, 11]
        delay_1 = random.choice(delay_choices_1)
        tik.sleep(delay_1)    
        news_time = int(time_list[i])
        if news_time <= trans_time and news_time > (trans_time - 86400):
            timestamp = str(datetime.fromtimestamp(news_time))
            print("Local-Date-time : " + str(timestamp) )
            print("UNIX-Timestamp : " + str(news_time))
            url_list = []
   
            get_news_link(i)
def get_news_link(j):
    global url
    global title
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'cookie' : 'nn_uid=ID=20230113124108:20031; uky=c_101-52-51-53-55-52-53-55; __gads=ID=a8f508913da9aae2:T=1673613768:S=ALNI_MaXhnKshaWNI3_B4S2tsL7ZrjUHyQ; _gid=GA1.3.1866699498.1674844904; cookie_policy=2; nnusrst=c_Pubs._QhZ-ArtList.1164452301;1164383991-CookieConsent.2; nn_sid=05f39165b0957402573c; __gpi=UID=00000ba35459f50e:T=1673613768:RT=1675190027:S=ALNI_MYgrQ3BB6jRaIPrz_4wPOGfjbi8mQ; nn_ssn=c_Qh; _ga=GA1.3.1553473696.1673583694; _ga_ZS7YYS67BF=GS1.1.1675180683.14.1.1675193984.0.0.0; NNNewsfeedHistory=c__C.326-A.Cyber+Security-K.Qu-S.1675194287-N.15653-Y.UKZ-_C.1-A.Technology-K.Qu-S.1675168527-N.15630-Y.UK; NN_Eng=0'
    }
    list_req = requests.get(target, headers=headers)
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
    print(content)
    Top = 0
    Popular = 0
    
    insert = \
            "INSERT INTO news_now (title,url,trans_url,response_status,post_date,timestamp,content,TOP,POPULAR) VALUES (%s, %s, %s, %s, %s, %s, %s,%s,%s)"
    data = (title, url,web,response.status_code,timestamp , news_time, content,Top,Popular)
    sql.execute(insert, data)
    conn.commit()
    content = ''

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--days", required = True,
                    help = "How many days do you want to search?",
                    type = int)
    args = parser.parse_args()
    print(f"How many days do you want to search?：至今前{args.days:^5}天，type={type(args.days)}")
    days = args.days
    amount = math.inf

    # # 清除資料庫
    # clear = str(input("Clear the DB ? y/n "))
    # if clear == 'y':
    #     command = "TRUNCATE TABLE news_now "
    #     sql.execute(command)
    #     print("Clear")

    # reset = str(input("Reset the A.I ? y/n "))
    # if reset == 'y':
    #     command_1 = "ALTER TABLE news_now Auto_Increment = 1"
    #     sql.execute(command_1)
    #     print("Done")
    for x in range(days):
        print("the time now is :" + str(datetime.fromtimestamp(trans_time)) + "/" + str(trans_time))
        print('-------------------------------------------------------')
        get_news_times()
        one_month_loop()


    delete = """
    DELETE FROM news_now WHERE postID NOT IN (SELECT MIN(postID) FROM news_now GROUP BY url)
    """
    sql.execute(delete)
    conn.commit()
    
    sql.close()
    conn.close()
    
  
    