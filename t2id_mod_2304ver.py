## 第一次執行本系統時，須執行以下指令
# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')

import pymysql
import pandas as pd
from bertopic import BERTopic
import configparser
config = configparser.ConfigParser()

def check_gpu():
    import torch
    print(torch.__version__)
    print(torch.cuda.is_available())

    import tensorflow as tf
    print(tf.test.gpu_device_name())

    return

def get_config(configfile):
    # Set config to connect DB server
    config.read(configfile)
    print("目前載入的Config檔為："+ str(config.read(configfile)))
    print("資料表：" + config['Command']['db_table'])

def latest_data_time():
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

    # 過濾 thecyberwire、bloomberg、computerweekly、databreach
    date = "" 
    try:
        # 建立Connection物件
        conn = pymysql.connect(**db_settings)
        deleted_count = 0
        # 建立Cursor物件
        cursor = conn.cursor()
        # 執行刪除操作
        cursor.execute("""
                        SELECT post_date FROM newsnow_security ORDER BY post_date DESC LIMIT 1;
                    """)
        # 提取結果
        result = cursor.fetchone()
        date = result[0]  # 提取日期部分

        # 關閉 Cursor
        conn.commit()
        # 關閉連接
        conn.close()
        print(date)
    except Exception as ex:
        print(ex)
    return date

def data_remove_redundant():
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

    # 過濾 thecyberwire、bloomberg、computerweekly、databreach
    try:
        # 建立Connection物件
        conn = pymysql.connect(**db_settings)
        deleted_count = 0
        # 建立Cursor物件
        cursor = conn.cursor()
        # 執行刪除操作
        cursor.execute("""
                    DELETE FROM newsnow_security where trans_url LIKE "%databreach%";
                    """)
        deleted_count += cursor.rowcount

        cursor.execute("""
                    DELETE FROM newsnow_security where trans_url LIKE "%thecyberwire.com%";
                    """)
        deleted_count += cursor.rowcount


        cursor.execute("""
                    DELETE FROM newsnow_security where trans_url LIKE "%bloomberg.com%";
                    """)
        deleted_count += cursor.rowcount

        cursor.execute("""
                    DELETE FROM newsnow_security where trans_url LIKE "%computerweekly%";
                    """)
        deleted_count += cursor.rowcount

        # 提交 commit
        conn.commit()
        # 關閉連接
        conn.close()
        # 顯示刪除資訊
        print(f"總計刪除 {deleted_count} 筆")

    except Exception as ex:
        print(ex)

    return (f"總計刪除 {deleted_count} 筆")



# 爬取資料
def data_retrieve_all():

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
    # print(db_settings)

    try:
        # 建立Connection物件
        conn = pymysql.connect(**db_settings)

        # 建立Cursor物件
        with conn.cursor() as cursor:
            # 查詢資料SQL語法
            command = config['Command'].get('command')
            all_df = pd.read_sql_query(command,conn)
            print("資料庫總文章數：" + str(all_df.__len__()))
            cursor.close

    except Exception as ex:
        print(ex)

    # 去除標題空白，並去除換行，放入 corpus
    for i in range(len(all_df)):
        all_df['title'].iloc[i] = all_df['title'].iloc[i].strip().replace('\n',' ').replace('\r',' ')

    # 把 dataframe 中的 post_data 資料，轉為 datetime
    all_df['post_date'] = pd.to_datetime(all_df['post_date'])

    return all_df

# 爬取資料 by time
def data_retrieve_by_time(start_date, end_date):
    import datetime
    print("擷取 TIMESTAMP：" + start_date +"~" + end_date + " 的文章數")

    # 讓日期取值正確，寫出日期加一天。再還原成字串
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    end_date = (end_date+datetime.timedelta(days=+1)).strftime("%Y-%m-%d")
    
    all_df = data_retrieve_all()
    # 擷取時間資料
    mask = (all_df['post_date'] >= start_date) & (all_df['post_date'] <= end_date)
    df = all_df.loc[mask]

    return df

def topic_number(topic_model):
    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info.Topic != -1]
    print("目前共有" , len(topic_info) , "個主題")
    return

def calc_nutrition(df_c_tf_idf):
    nutrition = df_c_tf_idf.sum(axis=1)
    # nutrition.to_csv("./nutrition_lemma.csv")
    return nutrition

def calc_c_tf_idf(topic_model):
    feature_names = topic_model.vectorizer_model.get_feature_names()
    docs_index = [n for n in topic_model.get_topic_info()['Name']]
    # 看單一天的 c-TF-IDF Table
    c_tf_idf_scores = topic_model.c_tf_idf_
    df_c_tf_idf = pd.DataFrame(c_tf_idf_scores.T.todense(), index = feature_names, columns = docs_index)
    return df_c_tf_idf

def build_timestamp(docs, stopwords):
    ### 輸入：docs
    ### 輸出：topic model & nutrition

    # model_name = "all-MiniLM-L6-v2"；"all-mpnet-base-v2"
    model_name = "all-mpnet-base-v2"
    top_n_words = 30
    print("embedding model:" + model_name)
    print("降維方法:UMAP")
    print("分群模型:HDBSCAN")
    print("top_n_words = " + str(top_n_words))
    
    # 開始用 BERTopic 轉向量，選擇模型 - 需GPU
    from sentence_transformers import SentenceTransformer, util
    sentence_model = SentenceTransformer(model_name, device="cuda")

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer_model = CountVectorizer(ngram_range=(1, 1), stop_words=stopwords)

    topic_model = BERTopic(embedding_model=sentence_model,vectorizer_model=vectorizer_model, language="english",top_n_words=top_n_words , verbose=True)
    topics, _ = topic_model.fit_transform(docs)

    df_c_tf_idf = calc_c_tf_idf(topic_model)
    nutrition = calc_nutrition(df_c_tf_idf)

    # 取得主題數量
    topic_number(topic_model)

    return topic_model, nutrition


def topic_length_rm(topic_model):
    # 把 -1 topic，去除
    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info.Topic != -1]
    topic_length = len(topic_info)
    
    return topic_length


def calc_burst_topic(topic_model, burst_sorted):
    import math

    # 把 -1 topic，去除
    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info.Topic != -1]
    topic_burst = []

    # 相比前一個 timestamp ，目前的 timestamp 有哪些主題是新興的
    for i in range (len(topic_info)):
        # 算出每一個主題的新興程度
        topic_burst_temp = 0
        for key in dict(topic_model.get_topic(i)):
            if (key in burst_sorted):
                topic_burst_temp += burst_sorted[key]
        topic_burst.append(topic_burst_temp)
        # print("第" +str(i) +"個主題："+ str(topic_burst_temp))
        
    topic_info['topic_burst'] = topic_burst
    return round(topic_info,2)

def calc_burst_doc(doc, burst_sorted):
    import nltk, time
    from nltk import word_tokenize
    import math

    word_set = set()  # 建立一個 set 來記錄已經出現過的單字
    doc_burst = 0  # 初始化 doc_burst 為 0

    for word in word_tokenize(doc):
        if word in burst_sorted and word not in word_set:  # 如果單字在 burst_sorted 中且還沒被計算過
            word_set.add(word)  # 把單字加入 set 中
            doc_burst += burst_sorted[word]  # 累加 doc_burst

    count = len(word_set)+1  # 計算單字數量，即為不重複的單字數量
    # print(count)
    return round(doc_burst/count,2)

def calc_burst_doc_2(doc, burst_sorted):
    import nltk, time
    from nltk import word_tokenize
    import math

    word_set = set()  # 建立一個 set 來記錄已經出現過的單字
    doc_burst = 0  # 初始化 doc_burst 為 0

    for word in word_tokenize(doc):
        if word in burst_sorted and word not in word_set:  # 如果單字在 burst_sorted 中且還沒被計算過
            word_set.add(word)  # 把單字加入 set 中
            doc_burst += burst_sorted[word]  # 累加 doc_burst

    count = len(word_set)+1  # 計算單字數量，即為不重複的單字數量
    # print(count)
    return round(doc_burst,2)


def plus_predict_label(topic_model, df):
    # 在 df 加上每個 topic 的 lable
    topic_label = topic_model.topics_
    df['predict_label'] = topic_label
    return df

def build_topic_info_with_burst(nutrition0, nutrition1, topic_model1):
    ### 輸入：nutrition0, nutrition1
    ### 輸出：burst，目前的計算公式是，有這個字就全放
    t0_nutr = nutrition0.to_dict()
    t1_nutr = nutrition1.to_dict()

    from tdt.algorithms import ELD, SlidingELD
    from tdt.nutrition.memory import MemoryNutritionStore

    store = MemoryNutritionStore()
    algo = ELD(store)

    # 新增 timestamp = 10 時的 Nutrition 值
    store.add(10, t0_nutr)
    store.add(20, t1_nutr)

    # 計算 burst 並排序，burst分數存在 burst_sorted
    burst = algo.detect(store.get(20), until=20, min_burst= 0)
    burst_sorted = dict(sorted(burst.items(), key=lambda item: item[1], reverse=True))

    # 對每個 topic ，計算 topic 的 burst 分數
    # topic_info_with_burst：處理後留有所有的主題dataframe，並加上 topic_burst 欄位(以主題代表字彙計算)
    topic_info_with_burst = calc_burst_topic(topic_model1,burst_sorted)
    return topic_info_with_burst, burst_sorted

def build_result(df,len):
    # 依照每一個 predict_label 進行 groupby
    df1_groupby = df.groupby(['predict_label'])

    result_df = pd.DataFrame()
    # 如果有 df 有 burst 欄位
    if("doc_burst" in df):
        # 找出每個群集之中，最新興的文章，前X篇(留下 title、url、predict label)
        for i in range(len):
            topic = df1_groupby.get_group(i).sort_values(by=['doc_burst'], ascending=False)
            result_df = pd.concat([result_df, topic.filter(items=['post_date','title', 'trans_url', 'content', 'doc_burst', 'burst_words', 'predict_label','content_processed','all_processed_list','title_processed'])])
    else:
        result_df = pd.concat([result_df, df.filter(items=['post_date','title', 'trans_url', 'content', 'predict_label','content_processed','all_processed_list','title_processed'])])

    result_df['post_date'] = result_df['post_date'].apply(lambda x: x.strftime("%Y-%m-%d"))
    # 重新設定索引，從1開始，用 df.loc[index] 取值
    result_df = result_df.reset_index(drop=True)
    result_df = result_df.reset_index()
    return result_df

def build_doc_burst(df1, burst_sorted):
    print("Formula 1")

    # 計算每一篇文章的加總 burst
    doc_burst = []
    for index, row in list(df1.iterrows()): # 1
        doc_burst.append(calc_burst_doc(row['content_processed'],burst_sorted))
    df1['doc_burst'] = doc_burst
    return df1

def build_doc_burst_2(df1, burst_sorted):    
    print("Formula 2")

    # 計算每一篇文章的加總 burst
    doc_burst = []
    for index, row in list(df1.iterrows()): # 1
        doc_burst.append(calc_burst_doc_2(row['content_processed'],burst_sorted))
    df1['doc_burst'] = doc_burst
    return df1

def find_representative_doc_url(topic_model, df, topic_index):
    # 找到主題群集代表文章的url
    print ("主題"+ str(topic_index) +"\n")
    for index, content in enumerate(topic_model.representative_docs_[topic_index]):
        print(str(index+1) +". "+ df['title'].where(df['title_processed'] == content.split("\n\n")[0]).dropna().iloc[0])
        print(df['url'].where(df['title_processed'] == content.split("\n\n")[0]).dropna().iloc[0])
    return

def visualize_pic_download(topic_model):
    import os
    from datetime import datetime

    # 取得現在時間
    time_now = datetime.now().strftime("%Y-%m-%d-%HH-%MM-%SS")
    pic_path = "./images/"+ time_now

    if not os.path.exists("./images/"):
            os.mkdir("./images/")
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)

    # 把結果變為可視化的主題圖 — Visualize Topics
    try:
        topic_model.visualize_topics().write_image(pic_path+"/topics.png",format="png")
    except Exception as e:
        print("主題圖產生失敗，原因：" + str(e))

    try:
        topic_model.visualize_hierarchy(top_n_topics=50).write_image(pic_path + "/hierarchy.png",format="png")
    except Exception as e:
        print("階層圖產生失敗，原因：" + str(e))

    try:
        topic_model.visualize_barchart(top_n_topics=20).write_image(pic_path + "/barchart.png",format="png")
    except Exception as e:
        print("字彙圖產生失敗，原因：" + str(e))

    return pic_path

def result_df_download(predict_df0, predict_df1, pic_path):
    predict_df0.to_csv(pic_path + "/predict_df0.csv")
    predict_df1.to_csv(pic_path + "/predict_df1.csv")
    return

# 回傳所有代表文章
def df_representative_doc_url(topic_model, df):
    list_of_lists = []
    for i in range(topic_model.representative_docs_.__len__()):
        for index, content in enumerate(topic_model.representative_docs_[i]):
            title = str(index+1) +". "+ df['title'].where(df['title_processed'] == content.split("\n\n")[0]).dropna().iloc[0]
            url = df['url'].where(df['title_processed'] == content.split("\n\n")[0]).dropna().iloc[0]
            temp_list = [i,title,url]
            list_of_lists.append(temp_list)
    return pd.DataFrame(list_of_lists,columns=['index','title','url'])

def html_representative_doc_url(topic_model, df):
    import pandas as pd
    list_of_lists = []
    for i in range(topic_model.representative_docs_.__len__()):
        temp_str = ""
        for index, content in enumerate(topic_model.representative_docs_[i]):
            title = str(index+1) +". "+ df['title'].where(df['title_processed'] == content.split("\n\n")[0]).dropna().iloc[0]
            url = df['trans_url'].where(df['title_processed'] == content.split("\n\n")[0]).dropna().iloc[0]
            doc_burst_normalize = df['doc_burst_normalize'].where(df['title_processed'] == content.split("\n\n")[0]).dropna().iloc[0]
            temp_str = temp_str + ("<td><a href="+ url + "target='_blank'>"+ title +"</a>" + " Burst：" + str(round(doc_burst_normalize,2)) +"</td><br>")

        list_of_lists.append(temp_str)
    return list_of_lists

def html_trend_doc_url(topic_model, predict_df_new):
    predict_df_groupby = predict_df_new.groupby(['predict_label'])
    cluster_trend_doc = {}
    for i in range(predict_df_groupby.size().__len__()):
        sort_trend_df = predict_df_groupby.get_group(i).sort_values(by=['doc_burst_normalize'], ascending=False)
        trend_doc = list()
        for j in range(0,3):
            trend_doc.append(sort_trend_df.iloc[j]['title_processed'])
            # print(sort_trend_df.iloc[j]['title_processed'])
        cluster_trend_doc[i] = trend_doc

    import pandas as pd
    list_of_lists = []
    for i in range(cluster_trend_doc.__len__()):
        temp_str = ""
        for j in range (cluster_trend_doc[i].__len__()):
            trendy = "[0]"

            url = predict_df_new['trans_url'].where(predict_df_new['title_processed'] == cluster_trend_doc[i][j]).dropna().iloc[0]
            doc_burst_normalize = predict_df_new['doc_burst_normalize'].where(predict_df_new['title_processed'] == cluster_trend_doc[i][j].split("\n\n")[0]).dropna().iloc[0]
            if(round(doc_burst_normalize,2) >= 0.5 ):
                trendy = "[1]"
            title = trendy + " " + str(i) +"-" + str(j+1) + " " + cluster_trend_doc[i][j]

            temp_str = temp_str + ("<td><a href="+ url + " target='_blank'>"+ title +"</a>" +"</td><br>")

        list_of_lists.append(temp_str)
    return list_of_lists

# 計算 T2ID 指標
def t2id_weight_calc(topic_model, predict_df):
    topic_keyword_weight = topic_model.get_topics()
    # 使用 dict comprehension 來篩選字典中不包含 -1 key 的項目
    topic_keyword_weight = {key: value for key, value in topic_keyword_weight.items() if key != -1}

    # t2id指標計算公式: doc_burst + 主題字彙權重*2
    # 每一個群集，計算所有的當篇文章有沒有包含這個字
    predict_df_groupby = predict_df.groupby(['predict_label'])
    t2id_weight_score = []
    for i in range(predict_df_groupby.size().__len__()):
        for index, row in list(predict_df_groupby.get_group(i).iterrows()): # 1
                temp_weight = row['doc_burst']
                for j in range(row['all_processed_list'].__len__()):
                    for k in range(topic_keyword_weight.get(i).__len__()):
                        if (row['all_processed_list'][j] == topic_keyword_weight.get(i)[k][0]):
                            temp_weight += (topic_keyword_weight.get(i)[k][1]*2)
                t2id_weight_score.append(round(temp_weight,2))

    predict_df = predict_df.sort_values(by=['index'])
    predict_df['t2id_weight_score'] = t2id_weight_score
    return predict_df

def data_preprocessing_lemma(df):
    ###--------------------去除重複值、lemma content、內文放入標題、Steming stopwords-----------------###

    # 去除 url 相同的重複值，by column data
    print("刪除前：" + str(df.shape))
    df = df.drop_duplicates(subset=["url"], keep='first')
    print("刪除後：" + str(df.shape))

    import nltk, time
    import numpy
    from nltk import word_tokenize
    from nltk.stem import WordNetLemmatizer
    import pandas as pd
    import re

    # Stemming，並放入標題
    corpus_processed = []
    title_processed = []
    all_processed_list = []

    wordnet_lemmatizer = WordNetLemmatizer()
    print("NLTK：Lemmatizer")

    # Stemming 標題
    for index, row in list(df.iterrows()): # 1
        word_vector_title = []
        row['title'] = row['title'].strip() # 去除掉多餘的空白
        title_list = word_tokenize(row['title'])
        all_processed_list_temp = []
        for i in range(title_list.__len__()):
            
            title_list[i] = title_list[i].lower()
            title_list[i] = wordnet_lemmatizer.lemmatize(title_list[i]) # 7
            all_processed_list_temp.append(title_list[i])
            word_vector_title.append(title_list[i])
        title_processed.append(' '.join(word_vector_title))
        all_processed_list.append(all_processed_list_temp)
    df['title_processed'] = title_processed

    # Stemming 內文，並放入標題(stemming 後的標題)
    corpus_processed = []
    count = 0
    for index, row in list(df.iterrows()): # 1
        word_vector = []
        content_list = word_tokenize(row['content'])
        all_processed_list_temp = []
        for i in range(content_list.__len__()):
            # content_list[i] = re.sub(r'\d+', '', content_list[i])

            content_list[i] = content_list[i].lower()
            content_list[i] = wordnet_lemmatizer.lemmatize(content_list[i]) # 7
            word_vector.append(content_list[i])
            all_processed_list_temp.append(content_list[i])
        corpus_processed.append(row['title_processed'] + "\n\n" + ' '.join(word_vector)) # 4
        all_processed_list[count].append(all_processed_list_temp)
        count += 1

    df['content_processed'] = corpus_processed
    df['all_processed_list'] = all_processed_list

    # Lemma stopword_t2id

    # 取得T2ID停用字 (包含俊傑+nltk)
    with open('./stopword_t2id', encoding='utf8') as f:
        stopword_t2id = f.readlines()
    stopwords = list(map(lambda s: wordnet_lemmatizer.lemmatize(s.strip()), stopword_t2id))

    return df, stopwords

def calc_tc_td(topic_model, docs):
    import gensim.corpora as corpora
    from gensim.models import KeyedVectors

    keys = topic_model.vectorizer_model.get_feature_names()
    vectors = topic_model.embedding_model.embed_words(topic_model.vectorizer_model.get_feature_names(),verbose= True)
    # 創建一個空的 KeyedVectors
    word_vectors = KeyedVectors(vector_size=768)
    # 向KeyedVectors對象添加詞向量
    word_vectors.add_vectors(keys, vectors)
    # 可選：將KeyedVectors對象保存到硬碟
    # word_vectors.save("word_vectors.kv")


    topics = topic_model.topics_
    # 取得乾淨，前處理後的資料
    documents = pd.DataFrame({"Document": docs,
                            "ID": range(len(docs)),
                            "Topic": topics})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)

    ###---------------------------------########
    # Extract vectorizer and tokenizer from BERTopic
    vectorizer = topic_model.vectorizer_model
    tokenizer = vectorizer.build_analyzer()

    # Extract features for Topic Coherence evaluation
    words = vectorizer.get_feature_names()
    tokens = [tokenizer(doc) for doc in cleaned_docs]

    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    # Extract words in each topic if they are non-empty and exist in the dictionary
    topic_words = []
    for topic in range(len(set(topics))-topic_model._outliers):
        words = list(zip(*topic_model.get_topic(topic)))[0]
        words = [word for word in words if word in dictionary.token2id]
        topic_words.append(words)
    topic_words = [words for words in topic_words if len(words) > 0]

    from gensim.models.coherencemodel import CoherenceModel
    # Evaluate
    coherence_model = CoherenceModel(topics=topic_words,
                                    texts=tokens,
                                    corpus=corpus,
                                    dictionary=dictionary,
                                    coherence='c_v')
    tc = coherence_model.get_coherence()
    
    import diversity_metrics
    # https://github.com/silviatti/topic-model-diversity
    td = diversity_metrics.centroid_distance(topic_words, word_vectors, topk=10)
    print("topic coherence: " + str(tc))
    print("topic diversity: " + str(td))
    return 

def build_burst_word(doc, burst_sorted):
    import nltk, time
    from nltk import word_tokenize
    import math
    # word_set = set()  # 建立一個 set 來記錄已經出現過的單字
    # burst_word = []  # 初始化 doc_burst 為 0

    # for word in word_tokenize(doc):
    #     if word in burst_sorted and word not in word_set    :  # 如果單字在 burst_sorted 中且還沒被計算過
    #         word_set.add(word)  # 把單字加入 set 中
    #         if(burst_sorted[word]>0.05):
    #             burst_word.append(word)  # 累加 doc_burst
    # return burst_word

    word_set = set()  # 建立一個 set 來記錄已經出現過的單字
    burst_word = dict()
    selected_burst_word = []

    for word in word_tokenize(doc):
        if word in burst_sorted and word not in word_set:  # 如果單字在 burst_sorted 中且還沒被計算過
            word_set.add(word)  # 把單字加入 set 中
            if(burst_sorted[word]>0):
                burst_word[word] = burst_sorted[word]  # 累加 doc_burst
    burst_word = dict(sorted(burst_word.items(), key=lambda item: item[1], reverse=True))
    # 最多取前五個
    selected_burst_word = list(burst_word.keys())[:5]
    return selected_burst_word


def build_burst_word_list(df1, burst_sorted):
    # 列出每一篇文章的 burst 字彙
    burst_words = []
    for index, row in list(df1.iterrows()): # 1
        burst_words.append(build_burst_word(row['content_processed'],burst_sorted))
    df1['burst_words'] = burst_words
    return df1

def exp7_build_topic_info_with_burst(nutr0, nutr1,nutr2,nutr3,nutr4,nutr5,nutr6,nutr7, topic_model):
    ### 輸入：nutrition0, nutrition1
    ### 輸出：burst，目前的計算公式是，有這個字就全放

    from tdt.algorithms import ELD, SlidingELD
    from tdt.nutrition.memory import MemoryNutritionStore

    store = MemoryNutritionStore()
    algo = ELD(store)

    # 新增 timestamp = 10 時的 Nutrition 值
    store.add(10, nutr0.to_dict())
    store.add(20, nutr1.to_dict())
    store.add(30, nutr2.to_dict())
    store.add(40, nutr3.to_dict())
    store.add(50, nutr4.to_dict())
    store.add(60, nutr5.to_dict())
    store.add(70, nutr6.to_dict())
    store.add(80, nutr7.to_dict())

    # 計算 burst 並排序，burst分數存在 burst_sorted
    burst = algo.detect(store.get(80), until=80, min_burst= 0)
    burst_sorted = dict(sorted(burst.items(), key=lambda item: item[1], reverse=True))

    # 對每個 topic ，計算 topic 的 burst 分數
    # topic_info_with_burst：處理後留有所有的主題dataframe，並加上 topic_burst 欄位(以主題代表字彙計算)
    topic_info_with_burst = calc_burst_topic(topic_model,burst_sorted)
    return topic_info_with_burst, burst_sorted

def exp3_build_topic_info_with_burst(nutr0, nutr1,nutr2,nutr3, topic_model):
    ### 輸入：nutrition0, nutrition1
    ### 輸出：burst，目前的計算公式是，有這個字就全放

    from tdt.algorithms import ELD, SlidingELD
    from tdt.nutrition.memory import MemoryNutritionStore

    store = MemoryNutritionStore()
    algo = ELD(store)

    # 新增 timestamp = 10 時的 Nutrition 值
    store.add(10, nutr0.to_dict())
    store.add(20, nutr1.to_dict())
    store.add(30, nutr2.to_dict())
    store.add(40, nutr3.to_dict())

    # 計算 burst 並排序，burst分數存在 burst_sorted
    burst = algo.detect(store.get(40), until=40, min_burst= 0)
    burst_sorted = dict(sorted(burst.items(), key=lambda item: item[1], reverse=True))

    # 對每個 topic ，計算 topic 的 burst 分數
    # topic_info_with_burst：處理後留有所有的主題dataframe，並加上 topic_burst 欄位(以主題代表字彙計算)
    topic_info_with_burst = calc_burst_topic(topic_model,burst_sorted)
    return topic_info_with_burst, burst_sorted

def exp2_build_topic_info_with_burst(nutr0, nutr1,nutr2, topic_model):
    ### 輸入：nutrition0, nutrition1
    ### 輸出：burst，目前的計算公式是，有這個字就全放

    from tdt.algorithms import ELD, SlidingELD
    from tdt.nutrition.memory import MemoryNutritionStore

    store = MemoryNutritionStore()
    algo = ELD(store)

    # 新增 timestamp = 10 時的 Nutrition 值
    store.add(10, nutr0.to_dict())
    store.add(20, nutr1.to_dict())
    store.add(30, nutr2.to_dict())

    # 計算 burst 並排序，burst分數存在 burst_sorted
    burst = algo.detect(store.get(30), until=30, min_burst= 0)
    burst_sorted = dict(sorted(burst.items(), key=lambda item: item[1], reverse=True))

    # 對每個 topic ，計算 topic 的 burst 分數
    # topic_info_with_burst：處理後留有所有的主題dataframe，並加上 topic_burst 欄位(以主題代表字彙計算)
    topic_info_with_burst = calc_burst_topic(topic_model,burst_sorted)
    return topic_info_with_burst, burst_sorted

def exp4_build_topic_info_with_burst(nutr0, nutr1,nutr2,nutr3,nutr4, topic_model):
    ### 輸入：nutrition0, nutrition1
    ### 輸出：burst，目前的計算公式是，有這個字就全放

    from tdt.algorithms import ELD, SlidingELD
    from tdt.nutrition.memory import MemoryNutritionStore

    store = MemoryNutritionStore()
    algo = ELD(store)

    # 新增 timestamp = 10 時的 Nutrition 值
    store.add(10, nutr0.to_dict())
    store.add(20, nutr1.to_dict())
    store.add(30, nutr2.to_dict())
    store.add(40, nutr3.to_dict())
    store.add(50, nutr4.to_dict())

    # 計算 burst 並排序，burst分數存在 burst_sorted
    burst = algo.detect(store.get(50), until=50, min_burst= 0)
    burst_sorted = dict(sorted(burst.items(), key=lambda item: item[1], reverse=True))

    # 對每個 topic ，計算 topic 的 burst 分數
    # topic_info_with_burst：處理後留有所有的主題dataframe，並加上 topic_burst 欄位(以主題代表字彙計算)
    topic_info_with_burst = calc_burst_topic(topic_model,burst_sorted)
    return topic_info_with_burst, burst_sorted

def view_df_burst_sorted(burst_sorted):
    import pandas as pd
    df = pd.DataFrame.from_dict(burst_sorted,columns=['values'],orient='index')
    df.reset_index(inplace=True)
    return df

def critical_drop(delta,burst_sorted):
    # 去除純數字的key
    burst_sorted = {k: v for k, v in burst_sorted.items() if not k.isdigit()}

    # user-driven drop
    burst_sum = sum(burst_sorted.values())
    drop = delta * (burst_sum/burst_sorted.__len__())
    print("drop:"+str(drop))
    print("drop前：" + str(burst_sorted.__len__()))
    df_burst_sorted = view_df_burst_sorted(burst_sorted)
    # Critical drop 後的 df
    df_burst_sorted_dropped = df_burst_sorted[0:df_burst_sorted[df_burst_sorted['values'] > drop].__len__()]
    print("drop後：" + str(df_burst_sorted_dropped.__len__()))

    # 轉成 dict
    burst_sorted_dropped = df_burst_sorted_dropped.set_index('index')['values'].to_dict()
    return burst_sorted_dropped