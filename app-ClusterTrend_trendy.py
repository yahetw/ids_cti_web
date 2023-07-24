from flask import Flask, render_template, request, jsonify, json, send_from_directory, abort
from flask_socketio import SocketIO, emit
import os 
import pandas as pd
import datetime
from dateutil.relativedelta import *
import configparser
config_host = configparser.ConfigParser()
config_host.read("./newsnow_config.ini")

###-------------------- 主程式--------------------###

import t2id_mod_2304ver as t2id
# 設定 config 檔案(必要)
t2id.get_config("./newsnow_config.ini")
app = Flask(__name__)
socketio = SocketIO(app)
###--flask 自動 reload 的cmd指令：flask run --reload---###

###--------------choose cpu or gpu--------------###

def check_gpu():
    global device

    import torch
    print(torch.__version__)
    print(torch.cuda.is_available())

    import tensorflow as tf
    print(tf.test.gpu_device_name())

    if(torch.cuda.is_available()):
        device = 'cuda'

    return

###---------------------宣告--------------------###

exp_time = None
df= t2id.data_retrieve_all()
predict_df = pd.DataFrame()
topic_info_with_burst = pd.DataFrame()
pic_path = None
count_list = [0, 0, 0, 0]
device = 'cpu'  # 預設使用cpu跑
check_gpu()

###---------------------count_list-------------###

all_df = df
### 將['post_date']欄位更改為 datetime 屬性
if isinstance(all_df['post_date'][0],str):
    all_df['post_date'] = pd.to_datetime(all_df['post_date'])

# 當日新聞數量
mask = (all_df['post_date'] >= (datetime.datetime.now()+datetime.timedelta(days=-1))) & (all_df['post_date'] <= (datetime.datetime.now()+datetime.timedelta(days=+1)))
df_today = all_df.loc[mask]
today_count = df_today.__len__()

# 當周新聞數量
mask = (all_df['post_date'] >= (datetime.datetime.now()+datetime.timedelta(weeks=-1))) & (all_df['post_date'] <= (datetime.datetime.now()+datetime.timedelta(days=+1)))
df_week = all_df.loc[mask]
week_count = df_week.__len__()

# 當月新聞數量
mask = (all_df['post_date'] >= (datetime.datetime.now()+relativedelta (months=-1))) & (all_df['post_date'] <= (datetime.datetime.now()+datetime.timedelta(days=+1)))
df_month = all_df.loc[mask]
month_count = df_month.__len__()

# 資料庫資料數量
db_data_count = all_df.__len__()
count_list = [today_count, week_count, month_count, db_data_count]

###---------------------限制存取-------------------###
trusted_ip = ['127.0.0.1', '140.117.80.142', '140.117.72.32']

# 孫杰：140.117.80.142
# 怡靜：140.117.172.32
@app.before_request
def limit_remote_addr():
    # print('請求網址：' + request.remote_addr)
    while request.remote_addr not in trusted_ip:
        abort(403) # Forbidden

###--------------------更新按鈕--------------------###

@app.route('/update_date', methods=['POST'])
def update_date():
    all_df = t2id.data_retrieve_all()
    ### 將['post_date']欄位更改為 datetime 屬性
    if isinstance(all_df['post_date'][0],str):
        all_df['post_date'] = pd.to_datetime(all_df['post_date'])

    # 當日新聞數量
    mask = (all_df['post_date'] >= (datetime.datetime.now()+datetime.timedelta(days=-1))) & (all_df['post_date'] <= (datetime.datetime.now()+datetime.timedelta(days=+1)))
    df_today = all_df.loc[mask]
    today_count = df_today.__len__()

    # 當周新聞數量
    mask = (all_df['post_date'] >= (datetime.datetime.now()+datetime.timedelta(weeks=-1))) & (all_df['post_date'] <= (datetime.datetime.now()+datetime.timedelta(days=+1)))
    df_week = all_df.loc[mask]
    week_count = df_week.__len__()

    # 當月新聞數量
    mask = (all_df['post_date'] >= (datetime.datetime.now()+relativedelta (months=-1))) & (all_df['post_date'] <= (datetime.datetime.now()+datetime.timedelta(days=+1)))
    df_month = all_df.loc[mask]
    month_count = df_month.__len__()
    
    # 資料庫資料數量
    db_data_count = all_df.__len__()
    count_list = [today_count, week_count, month_count, db_data_count]
    
    return render_template('index.html',count_list = count_list,  latest_data_time = t2id.latest_data_time())



@app.route('/refresh', methods=['POST'])
def refresh_date():
    all_df = t2id.data_retrieve_all()
    ### 將['post_date']欄位更改為 datetime 屬性
    if isinstance(all_df['post_date'][0],str):
        all_df['post_date'] = pd.to_datetime(all_df['post_date'])

    # 當日新聞數量
    mask = (all_df['post_date'] >= (datetime.datetime.now()+datetime.timedelta(days=-1))) & (all_df['post_date'] <= (datetime.datetime.now()+datetime.timedelta(days=+1)))
    df_today = all_df.loc[mask]
    today_count = df_today.__len__()

    # 當周新聞數量
    mask = (all_df['post_date'] >= (datetime.datetime.now()+datetime.timedelta(weeks=-1))) & (all_df['post_date'] <= (datetime.datetime.now()+datetime.timedelta(days=+1)))
    df_week = all_df.loc[mask]
    week_count = df_week.__len__()

    # 當月新聞數量
    mask = (all_df['post_date'] >= (datetime.datetime.now()+relativedelta (months=-1))) & (all_df['post_date'] <= (datetime.datetime.now()+datetime.timedelta(days=+1)))
    df_month = all_df.loc[mask]
    month_count = df_month.__len__()
    
    # 資料庫資料數量
    db_data_count = all_df.__len__()
    count_list = [today_count, week_count, month_count, db_data_count]
    
    global df
    t2id.data_remove_redundant()
    df = t2id.data_retrieve_all()
    print("資料庫 & 網站更新成功")
    return render_template('index.html',count_list = count_list,  latest_data_time = t2id.latest_data_time())


###---------------------主程式--------------------###

def h12c3(time_start):
    global pic_path
    global topic_info_with_burst
    t2id.data_remove_redundant()
    time_start = datetime.datetime.strptime(time_start,'%Y-%m-%d')
    # H12 C3
    time_0_start = (time_start+datetime.timedelta(days=-15)).strftime("%Y-%m-%d")
    time_0_end = (time_start+datetime.timedelta(days=-13)).strftime("%Y-%m-%d")
    time_1_start = (time_start+datetime.timedelta(days=-12)).strftime("%Y-%m-%d")
    time_1_end = (time_start+datetime.timedelta(days=-10)).strftime("%Y-%m-%d")
    time_2_start = (time_start+datetime.timedelta(days=-9)).strftime("%Y-%m-%d")
    time_2_end = (time_start+datetime.timedelta(days=-7)).strftime("%Y-%m-%d")
    time_3_start = (time_start+datetime.timedelta(days=-6)).strftime("%Y-%m-%d")
    time_3_end = (time_start+datetime.timedelta(days=-4)).strftime("%Y-%m-%d")
    time_4_start = (time_start+datetime.timedelta(days=-3)).strftime("%Y-%m-%d")
    time_4_end = (time_start).strftime("%Y-%m-%d")

    # 建立 timestamp，替每個 timestamp 分別建立一個模型
    df0 = t2id.data_retrieve_by_time(time_0_start, time_0_end)
    df0, stopwords = t2id.data_preprocessing_lemma(df0)
    df1 = t2id.data_retrieve_by_time(time_1_start, time_1_end)
    df1, stopwords = t2id.data_preprocessing_lemma(df1)
    df2 = t2id.data_retrieve_by_time(time_2_start, time_2_end)
    df2, stopwords = t2id.data_preprocessing_lemma(df2)
    df3 = t2id.data_retrieve_by_time(time_3_start, time_3_end)
    df3, stopwords = t2id.data_preprocessing_lemma(df3)
    df_new = t2id.data_retrieve_by_time(time_4_start, time_4_end)
    df_new, stopwords = t2id.data_preprocessing_lemma(df_new)

    docs0 = df0['content_processed'].tolist()
    docs1 = df1['content_processed'].tolist()
    docs2 = df2['content_processed'].tolist()
    docs3 = df3['content_processed'].tolist()
    docs_new = df_new['content_processed'].tolist()

    topic_model0, nutr0 = t2id.build_timestamp(docs0, stopwords,device)
    topic_model1, nutr1 = t2id.build_timestamp(docs1, stopwords,device)
    topic_model2, nutr2 = t2id.build_timestamp(docs2, stopwords,device)
    topic_model3, nutr3 = t2id.build_timestamp(docs3, stopwords,device)
    topic_model_new, nutr_new = t2id.build_timestamp(docs_new, stopwords,device)

    # 計算 timestamp1 相比 timestamp0 的 burst
    topic_info_with_burst, burst_sorted = t2id.exp4_build_topic_info_with_burst(nutr0, nutr1, nutr2, nutr3,nutr_new, topic_model_new)
    # Min-Max
    topic_info_with_burst['topic_burst_normalize'] = (topic_info_with_burst['topic_burst'] - topic_info_with_burst['topic_burst'].min()) / (topic_info_with_burst['topic_burst'].max() - topic_info_with_burst['topic_burst'].min())
    topic_info_with_burst['topic_burst'] = round(topic_info_with_burst['topic_burst_normalize'],2)

    df_new = t2id.plus_predict_label(topic_model_new, df_new)

    # critical_drop
    burst_sorted_dropped = t2id.critical_drop(10,burst_sorted)
    # 替每篇文章加上 burst 值
    df_new = t2id.build_doc_burst_2(df_new, burst_sorted_dropped)
    df_new = t2id.build_burst_word_list(df_new, burst_sorted_dropped)
    len = topic_info_with_burst.__len__()

    # result，並依照 doc_burst 排序，只有 df1 有
    predict_df_new = t2id.build_result(df_new,len)
    predict_df_new['doc_burst_normalize'] = (predict_df_new['doc_burst'] - predict_df_new['doc_burst'].min()) / (predict_df_new['doc_burst'].max() - predict_df_new['doc_burst'].min())
    predict_df_new['doc_burst'] = round(predict_df_new['doc_burst_normalize'],2)

    pic_path = t2id.visualize_pic_download(topic_model_new, predict_df_new)
    t2id.result_df_download(df0, predict_df_new, pic_path)
    topic_info_with_burst['repr_url'] = t2id.html_trend_doc_url(topic_model_new, predict_df_new)

    return predict_df_new

def h9c3(time_start):
    global pic_path
    global topic_info_with_burst
    t2id.data_remove_redundant()
    time_start = datetime.datetime.strptime(time_start,'%Y-%m-%d')
    # H9 C3
    time_0_start = (time_start+datetime.timedelta(days=-12)).strftime("%Y-%m-%d")
    time_0_end = (time_start+datetime.timedelta(days=-10)).strftime("%Y-%m-%d")
    time_1_start = (time_start+datetime.timedelta(days=-9)).strftime("%Y-%m-%d")
    time_1_end = (time_start+datetime.timedelta(days=-7)).strftime("%Y-%m-%d")
    time_2_start = (time_start+datetime.timedelta(days=-6)).strftime("%Y-%m-%d")
    time_2_end = (time_start+datetime.timedelta(days=-4)).strftime("%Y-%m-%d")
    time_3_start = (time_start+datetime.timedelta(days=-3)).strftime("%Y-%m-%d")
    time_3_end = (time_start).strftime("%Y-%m-%d")

    # 建立 timestamp，替每個 timestamp 分別建立一個模型
    df0 = t2id.data_retrieve_by_time(time_0_start, time_0_end)
    df0, stopwords = t2id.data_preprocessing_lemma(df0)
    df1 = t2id.data_retrieve_by_time(time_1_start, time_1_end)
    df1, stopwords = t2id.data_preprocessing_lemma(df1)
    df2 = t2id.data_retrieve_by_time(time_2_start, time_2_end)
    df2, stopwords = t2id.data_preprocessing_lemma(df2)
    df_new = t2id.data_retrieve_by_time(time_3_start, time_3_end)
    df_new, stopwords = t2id.data_preprocessing_lemma(df_new)

    docs0 = df0['content_processed'].tolist()
    docs1 = df1['content_processed'].tolist()
    docs2 = df2['content_processed'].tolist()
    docs_new = df_new['content_processed'].tolist()

    topic_model0, nutr0 = t2id.build_timestamp(docs0, stopwords,device)
    topic_model1, nutr1 = t2id.build_timestamp(docs1, stopwords,device)
    topic_model2, nutr2 = t2id.build_timestamp(docs2, stopwords,device)
    topic_model_new, nutr_new = t2id.build_timestamp(docs_new, stopwords,device)

    # 計算 timestamp1 相比 timestamp0 的 burst
    topic_info_with_burst, burst_sorted = t2id.exp3_build_topic_info_with_burst(nutr0, nutr1, nutr2,nutr_new, topic_model_new)
    # Min-Max
    topic_info_with_burst['topic_burst_normalize'] = (topic_info_with_burst['topic_burst'] - topic_info_with_burst['topic_burst'].min()) / (topic_info_with_burst['topic_burst'].max() - topic_info_with_burst['topic_burst'].min())
    topic_info_with_burst['topic_burst'] = round(topic_info_with_burst['topic_burst_normalize'],2)

    df_new = t2id.plus_predict_label(topic_model_new, df_new)

    # critical_drop
    burst_sorted_dropped = t2id.critical_drop(10,burst_sorted)
    # 替每篇文章加上 burst 值
    df_new = t2id.build_doc_burst_2(df_new, burst_sorted_dropped)
    df_new = t2id.build_burst_word_list(df_new, burst_sorted_dropped)
    len = topic_info_with_burst.__len__()

    # result，並依照 doc_burst 排序，只有 df1 有
    predict_df_new = t2id.build_result(df_new,len)
    predict_df_new['doc_burst_normalize'] = (predict_df_new['doc_burst'] - predict_df_new['doc_burst'].min()) / (predict_df_new['doc_burst'].max() - predict_df_new['doc_burst'].min())
    predict_df_new['doc_burst'] = round(predict_df_new['doc_burst_normalize'],2)

    pic_path = t2id.visualize_pic_download(topic_model_new,predict_df_new)
    t2id.result_df_download(df0, predict_df_new, pic_path)
    topic_info_with_burst['repr_url'] = t2id.html_trend_doc_url(topic_model_new, predict_df_new)

    return predict_df_new


def h3c1(time_start):
    global pic_path
    global topic_info_with_burst
    t2id.data_remove_redundant()
    time_start = datetime.datetime.strptime(time_start,'%Y-%m-%d')
    # H3 C1
    time_0_start = (time_start+datetime.timedelta(days=-3)).strftime("%Y-%m-%d")
    time_0_end = (time_start+datetime.timedelta(days=-3)).strftime("%Y-%m-%d")
    time_1_start = (time_start+datetime.timedelta(days=-2)).strftime("%Y-%m-%d")
    time_1_end = (time_start+datetime.timedelta(days=-2)).strftime("%Y-%m-%d")
    time_2_start = (time_start+datetime.timedelta(days=-1)).strftime("%Y-%m-%d")
    time_2_end = (time_start+datetime.timedelta(days=-1)).strftime("%Y-%m-%d")
    time_3_start = time_start.strftime("%Y-%m-%d")
    time_3_end = time_start.strftime("%Y-%m-%d")
    # 建立 timestamp，替每個 timestamp 分別建立一個模型
    df0 = t2id.data_retrieve_by_time(time_0_start, time_0_end)
    df0, stopwords = t2id.data_preprocessing_lemma(df0)
    df1 = t2id.data_retrieve_by_time(time_1_start, time_1_end)
    df1, stopwords = t2id.data_preprocessing_lemma(df1)
    df2 = t2id.data_retrieve_by_time(time_2_start, time_2_end)
    df2, stopwords = t2id.data_preprocessing_lemma(df2)
    df_new = t2id.data_retrieve_by_time(time_3_start, time_3_end)
    df_new, stopwords = t2id.data_preprocessing_lemma(df_new)

    docs0 = df0['content_processed'].tolist()
    docs1 = df1['content_processed'].tolist()
    docs2 = df2['content_processed'].tolist()
    docs_new = df_new['content_processed'].tolist()

    topic_model0, nutr0 = t2id.build_timestamp(docs0, stopwords,device)
    topic_model1, nutr1 = t2id.build_timestamp(docs1, stopwords,device)
    topic_model2, nutr2 = t2id.build_timestamp(docs2, stopwords,device)
    topic_model_new, nutr_new = t2id.build_timestamp(docs_new, stopwords,device)

    # 計算 timestamp1 相比 timestamp0 的 burst
    topic_info_with_burst, burst_sorted = t2id.exp3_build_topic_info_with_burst(nutr0, nutr1, nutr2,nutr_new, topic_model_new)
    # Min-Max
    topic_info_with_burst['topic_burst_normalize'] = (topic_info_with_burst['topic_burst'] - topic_info_with_burst['topic_burst'].min()) / (topic_info_with_burst['topic_burst'].max() - topic_info_with_burst['topic_burst'].min())
    topic_info_with_burst['topic_burst'] = round(topic_info_with_burst['topic_burst_normalize'],2)

    df_new = t2id.plus_predict_label(topic_model_new, df_new)

    # critical_drop
    burst_sorted_dropped = t2id.critical_drop(10,burst_sorted)
    # 替每篇文章加上 burst 值
    df_new = t2id.build_doc_burst_2(df_new, burst_sorted_dropped)
    df_new = t2id.build_burst_word_list(df_new, burst_sorted_dropped)
    len = topic_info_with_burst.__len__()

    # result，並依照 doc_burst 排序，只有 df1 有
    predict_df_new = t2id.build_result(df_new,len)
    predict_df_new['doc_burst_normalize'] = (predict_df_new['doc_burst'] - predict_df_new['doc_burst'].min()) / (predict_df_new['doc_burst'].max() - predict_df_new['doc_burst'].min())
    predict_df_new['doc_burst'] = round(predict_df_new['doc_burst_normalize'],2)

    pic_path = t2id.visualize_pic_download(topic_model_new,predict_df_new)
    t2id.result_df_download(df0, predict_df_new, pic_path)
    topic_info_with_burst['repr_url'] = t2id.html_trend_doc_url(topic_model_new, predict_df_new)

    # if isinstance(predict_df_new['post_date'][0],str)!=1:
    #     predict_df_new['post_date'] = predict_df_new['post_date'].apply(lambda x: x.strftime("%Y-%m-%d"))
    # predict_df_show = predict_df_new.sort_values(by=['doc_burst_normalize'], ascending=False)
    # predict_df_show = predict_df_show.head(30)

    return predict_df_new

def h7c1(time_start):
    global pic_path
    global topic_info_with_burst
    t2id.data_remove_redundant()
    time_start = datetime.datetime.strptime(time_start,'%Y-%m-%d')
    # H7 C1
    time_0_start = (time_start+datetime.timedelta(days=-7)).strftime("%Y-%m-%d")
    time_0_end = (time_start+datetime.timedelta(days=-7)).strftime("%Y-%m-%d")
    time_1_start = (time_start+datetime.timedelta(days=-6)).strftime("%Y-%m-%d")
    time_1_end = (time_start+datetime.timedelta(days=-6)).strftime("%Y-%m-%d")
    time_2_start = (time_start+datetime.timedelta(days=-5)).strftime("%Y-%m-%d")
    time_2_end = (time_start+datetime.timedelta(days=-5)).strftime("%Y-%m-%d")
    time_3_start = (time_start+datetime.timedelta(days=-4)).strftime("%Y-%m-%d")
    time_3_end = (time_start+datetime.timedelta(days=-4)).strftime("%Y-%m-%d")
    time_4_start = (time_start+datetime.timedelta(days=-3)).strftime("%Y-%m-%d")
    time_4_end = (time_start+datetime.timedelta(days=-3)).strftime("%Y-%m-%d")
    time_5_start = (time_start+datetime.timedelta(days=-2)).strftime("%Y-%m-%d")
    time_5_end = (time_start+datetime.timedelta(days=-2)).strftime("%Y-%m-%d")
    time_6_start = (time_start+datetime.timedelta(days=-1)).strftime("%Y-%m-%d")
    time_6_end = (time_start+datetime.timedelta(days=-1)).strftime("%Y-%m-%d")
    time_7_start = time_start.strftime("%Y-%m-%d")
    time_7_end = time_start.strftime("%Y-%m-%d")

    # 建立 timestamp，替每個 timestamp 分別建立一個模型
    df0 = t2id.data_retrieve_by_time(time_0_start, time_0_end)
    df0, stopwords = t2id.data_preprocessing_lemma(df0)
    df1 = t2id.data_retrieve_by_time(time_1_start, time_1_end)
    df1, stopwords = t2id.data_preprocessing_lemma(df1)
    df2 = t2id.data_retrieve_by_time(time_2_start, time_2_end)
    df2, stopwords = t2id.data_preprocessing_lemma(df2)
    df3 = t2id.data_retrieve_by_time(time_3_start, time_3_end)
    df3, stopwords = t2id.data_preprocessing_lemma(df3)
    df4 = t2id.data_retrieve_by_time(time_4_start, time_4_end)
    df4, stopwords = t2id.data_preprocessing_lemma(df4)
    df5 = t2id.data_retrieve_by_time(time_5_start, time_5_end)
    df5, stopwords = t2id.data_preprocessing_lemma(df5)
    df6 = t2id.data_retrieve_by_time(time_6_start, time_6_end)
    df6, stopwords = t2id.data_preprocessing_lemma(df6)
    df_new = t2id.data_retrieve_by_time(time_7_start, time_7_end)
    df_new, stopwords = t2id.data_preprocessing_lemma(df_new)

    docs0 = df0['content_processed'].tolist()
    docs1 = df1['content_processed'].tolist()
    docs2 = df2['content_processed'].tolist()
    docs3 = df3['content_processed'].tolist()
    docs4 = df4['content_processed'].tolist()
    docs5 = df5['content_processed'].tolist()
    docs6 = df6['content_processed'].tolist()
    docs_new = df_new['content_processed'].tolist()

    topic_model0, nutr0 = t2id.build_timestamp(docs0, stopwords,device)
    topic_model1, nutr1 = t2id.build_timestamp(docs1, stopwords,device)
    topic_model2, nutr2 = t2id.build_timestamp(docs2, stopwords,device)
    topic_model3, nutr3 = t2id.build_timestamp(docs3, stopwords,device)
    topic_model4, nutr4 = t2id.build_timestamp(docs4, stopwords,device)
    topic_model5, nutr5 = t2id.build_timestamp(docs5, stopwords,device)
    topic_model6, nutr6 = t2id.build_timestamp(docs6, stopwords,device)
    topic_model_new, nutr_new = t2id.build_timestamp(docs_new, stopwords,device)

    # 計算 timestamp1 相比 timestamp0 的 burst
    topic_info_with_burst, burst_sorted = t2id.exp7_build_topic_info_with_burst(nutr0, nutr1, nutr2, nutr3, nutr4, nutr5, nutr6, nutr_new, topic_model_new)
    # Min-Max
    topic_info_with_burst['topic_burst_normalize'] = (topic_info_with_burst['topic_burst'] - topic_info_with_burst['topic_burst'].min()) / (topic_info_with_burst['topic_burst'].max() - topic_info_with_burst['topic_burst'].min())
    topic_info_with_burst['topic_burst'] = round(topic_info_with_burst['topic_burst_normalize'],2)

    df_new = t2id.plus_predict_label(topic_model_new, df_new)

    # critical_drop
    burst_sorted_dropped = t2id.critical_drop(10,burst_sorted)
    # 替每篇文章加上 burst 值
    df_new = t2id.build_doc_burst_2(df_new, burst_sorted_dropped)
    df_new = t2id.build_burst_word_list(df_new, burst_sorted_dropped)
    len = topic_info_with_burst.__len__()

    # result，並依照 doc_burst 排序，只有 df1 有
    predict_df_new = t2id.build_result(df_new,len)
    predict_df_new['doc_burst_normalize'] = (predict_df_new['doc_burst'] - predict_df_new['doc_burst'].min()) / (predict_df_new['doc_burst'].max() - predict_df_new['doc_burst'].min())
    predict_df_new['doc_burst'] = round(predict_df_new['doc_burst_normalize'],2)

    pic_path = t2id.visualize_pic_download(topic_model_new,predict_df_new)
    t2id.result_df_download(df0, predict_df_new, pic_path)
    topic_info_with_burst['repr_url'] = t2id.html_trend_doc_url(topic_model_new, predict_df_new)

    return predict_df_new

###----------------------view_all_article.html-----------###

@app.route("/view_all_article.html")
def view_all_article():
    global df
    df = df.filter(items=['post_date','title', 'trans_url','content'])
    df = df.reset_index(drop=True)
    df = df.reset_index()
    return render_template('view_all_article.html', df = df.values.tolist())

###-----------------------cti_view.html------------------###

@app.route("/cti_view.html")
def cti_view():
    # 如果有資料，則回傳上次設定參數的頁面
    if(not predict_df.empty):
        return render_template('cti_view.html', df=predict_df.values.tolist(), exp_time = exp_time, user_ip = request.remote_addr)
    else:
        return render_template('no_data.html')

@app.route("/no_data.html")
def no_data():
    return render_template('no_data.html')

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    import os
    from datetime import datetime
    time_now = datetime.now().strftime("%Y-%m-%d-%HH-%MM-%SS")

    # 接收上傳的 CSV 檔案
    file = request.files['file']
    # 讀取 CSV 檔案並轉換成 DataFrame
    df = pd.read_csv(file)
    predict_df_new['trendy'] = df['趨勢']

    # 儲存 CSV 檔案
    if not os.path.exists("./網頁標記"):
            os.mkdir("./網頁標記")

    predict_df_new.to_csv('./網頁標記/標記-'+ time_now +'，IP-'+ request.remote_addr + '.csv')
    return 'File uploaded successfully'

###----------------------------------------------------------###

# 首頁
@app.route("/", methods=['GET', 'POST'])
@app.route('/index.html', methods=['GET', 'POST'])
def index():
    global predict_df
    global exp_time
    if request.method == 'POST':
            if(request.form['date-input'] != ''):
                date_input = request.form['date-input']  # 獲取日期欄位
                date_input_last = date_input
                past_dates_last = (datetime.datetime.strptime(date_input,'%Y-%m-%d') - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                past_dates = (datetime.datetime.strptime(date_input,'%Y-%m-%d') - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
                predict_df = h7c1(date_input)

            if(request.form['date-input2'] != ''):
                date_input = request.form['date-input2']  # 獲取日期欄位
                date_input_last = date_input
                past_dates_last = (datetime.datetime.strptime(date_input,'%Y-%m-%d') - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                past_dates = (datetime.datetime.strptime(date_input,'%Y-%m-%d') - datetime.timedelta(days=3)).strftime('%Y-%m-%d')
                predict_df = h3c1(date_input)
            if(request.form['date-input3'] != ''):
                date_input = request.form['date-input3']  # 獲取日期欄位
                date_input_last = (datetime.datetime.strptime(date_input,'%Y-%m-%d') - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
                past_dates_last = (datetime.datetime.strptime(date_input,'%Y-%m-%d') - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                past_dates = (datetime.datetime.strptime(date_input,'%Y-%m-%d') - datetime.timedelta(days=9)).strftime('%Y-%m-%d')
                predict_df = h9c3(date_input)
            if(request.form['date-input4'] != ''):
                date_input = request.form['date-input4']  # 獲取日期欄位
                date_input_last = (datetime.datetime.strptime(date_input,'%Y-%m-%d') - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
                past_dates_last = (datetime.datetime.strptime(date_input,'%Y-%m-%d') - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                past_dates = (datetime.datetime.strptime(date_input,'%Y-%m-%d') - datetime.timedelta(days=12)).strftime('%Y-%m-%d')
                predict_df = h12c3(date_input)
            exp_time = [past_dates, past_dates_last, date_input_last, date_input]

            return render_template('cti_view.html', df=predict_df.values.tolist(), exp_time = exp_time, user_ip = request.remote_addr)

    return render_template('index.html',count_list = count_list,latest_data_time = t2id.latest_data_time())

# 設定路由(routing)，自動回應
@app.route('/data/<index>', methods=['GET'])
def queryData(index):
    # predict_df_new.loc[int(index)][i]
    return render_template("./cti_panel.html", df=predict_df.loc[int(index)].values.tolist())

@app.route('/data_all/<index>', methods=['GET'])
def queryAllData(index):
    # predict_df_new.loc[int(index)][i]
    return render_template("./cti_panel.html", df= df.loc[int(index)].values.tolist())

###--------------------Chart--------------------###

@app.route('/chart.html')
def chart():

    if(pic_path):
            return render_template("./chart.html",path = pic_path)
    else:
         return render_template("no_data.html",)

    # return send_file("./images/2023-02-19,17H-28M-20S"+"/topics.png")

@app.route('/get_image')
def get_image():
    # http://127.0.0.1:5000/get_image?path=./images/2023-02-19,17H-28M-20S&name=topics.png
    filename = request.args.get('name')
    pic_path = request.args.get('path')
    return send_from_directory(pic_path , filename, mimetype='image/png')

###---------------------T2ID Result--------------------###

@app.route('/selected_result.html')
def selected_result():
    if(not topic_info_with_burst.empty):
        return render_template("./selected_result.html",topic_info_with_burst=topic_info_with_burst.values.tolist(),path = pic_path)
    else:
        return render_template('no_data.html')
    

@app.route('/real_time_response.html')
def real_time_response():
    return render_template('real_time_response.html', latest_data_time = t2id.latest_data_time(), socket_host = config_host['socket_host']['ip'])


@socketio.on('execute_refreshDB')
def handle_execute_command():
    # 在這裡執行命令並回傳執行結果
    # 假設你有一個執行命令的函式，例如 execute_command(command)，可以自行實現這個函式

    # 呼叫執行命令的函式，並逐行回傳執行結果
    global df
    socketio.emit('command_output', "資料庫更新中...請稍後，需等待大約1分鐘，顯示成功訊息即可繼續!")
    socketio.emit('command_output', t2id.data_remove_redundant())
    df = t2id.data_retrieve_all()
    socketio.emit('command_output', "資料庫 & 網站更新成功")
    socketio.emit('command_output', "------------------------------------")
    # socketio.emit('command_output', output_line)

@socketio.on('execute_catch')
def handle_execute_command():
    socketio.emit('command_output', "爬取近期資料中...請稍後，大約20分鐘，顯示成功訊息即可繼續!")

    import subprocess
    # 指定 .bat 檔的路徑
    bat_file_path = config_host['bat_path']['path']
    print(bat_file_path)

    # 執行 .bat 檔
    result = subprocess.run(bat_file_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    socketio.emit('command_output',  bat_file_path)

    # 檢查執行結果
    if result.returncode == 0:
        output = result.stdout
        print(output)
        socketio.emit('command_output', "資料抓取成功")
    else:
        error = result.stderr
        print(error)
        socketio.emit('command_output', f"資料抓取失敗: {error}")
    socketio.emit('command_output', "------------------------------------")


##---------------------------------------------###
if __name__ == '__main__':
    # 執行 flask
    # 若要 auto-reloading，將 debug 設為 True
    socketio.run(app, host="0.0.0.0", port=8001, debug=False)