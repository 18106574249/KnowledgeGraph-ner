# -*- coding:UTF-8 -*-
import time
import random
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import sys
import re
import jieba
import jieba.analyse
import pymysql

def stop_words(path):
    with open(path, 'r', encoding='utf-8') as f:
        sw = []
        for line in f:
            sw.append(line.strip())

    return set(sw)


def word_cut_ch(line, sw):
    # 保留中文
    rule = re.compile("[^\u4e00-\u9fa5]")
    line = re.sub(rule, "", line)
    if sw is not None:
        line = " ".join([word for word in jieba.cut(line, cut_all=False, HMM=True) if word not in sw])

    return line


def time2unix(dt):
    try:
        timeArray = dt.split(":")
        sec = 0
        for t in timeArray:
            sec = sec * 60 + int(t)
        return sec * 1000
    except ValueError:
        return np.nan


def add_text(label_df):
    try:
        db = pymysql.connect("123.56.28.235", "ayf0402", "ayf0402", "xdf-resource", charset='utf8')
        cursor = db.cursor()

        sql = "SELECT CONVERT(SUBSTRING(CONVERT(file_id,CHAR),7),UNSIGNED),type,school_id,duration_time,bg,ed,model_check,batch_code from `xdf_tag_fragment` where status=3 and type is not null and model=2 and bg is not null and ed is not null and type in (2,1,3,11,12,14,13) and duration_time>0;"
        cursor.execute(sql)
        results = cursor.fetchall()
        # sqldb="SELECT id,db_sec from `xdf_lesson_info_201904` WHERE id in (select distinct CONVERT(SUBSTRING(CONVERT(file_id,CHAR),7),UNSIGNED) from `xdf_tag_fragment` where status=3 and type is not null and model=2 and bg is not null and ed is not null);"
        # cursor.execute(sqldb)
        # dbresults=cursor.fetchall()
    except Exception as e:
        print("MySQL error")

    data = pd.DataFrame(list(results),
                        columns=['id', 'type', 'school_id', 'duration', 'bg', 'ed', 'model_check', 'batch_code'])

    # dbdata=pd.DataFrame(list(dbresults),columns=['id','db_sec'])
    # dbdata = pd.read_excel('db_sec.xlsx', encoding='utf-8', index=False)
    # # print(str(dbdata.loc[0,'db_sec']).split(','))
    # dbdata['db_sec'] = dbdata['db_sec'].map(lambda x: str(x).split(","))  # db_sec是list，list内是str '99.225'
    # dblength = []
    # for i in dbdata['db_sec'].values:
    #     dblength.append(len(i))
    # plt.hist(dblength)
    # plt.show()
    print(data.describe())
    print(data['type'].value_counts())
    # label_text_df = pd.merge(data,label_df,how='inner',on="id")
    # label_text_df.drop('file_id',axis=1,inplace=True)
    label_text_df = pd.merge(label_df, data, how='inner', on='id')
    return label_text_df


def get_data(label_df, sw_path, type, school=None, shuffle=True):
    '''

    :param label_df: 未加字幕的dataframe
    :param sw_path: stopwords文件的路径
    :param type: str,选取哪种类型的数据 1：态度不好，2：纪律不好，3：闲聊
    :param shuffle:
    :return: data:ndarry of str 未分词的片段字幕
            data_cut:ndarry of list 分词取停词后的字幕
            label:标签,传统分类器是1,0
            max_seq_len:rnn用，每个序列的最长片段
    '''



    # 加入字幕
    label_text_df = add_text(label_df)
    if not school is None:
        if school == 'paopao':
            label_text_df = label_text_df[label_text_df['school_id']<=50]
        elif school == 'changping':
            label_text_df = label_text_df[label_text_df['school_id']==53]
        elif school == 'lanzhou':
            label_text_df = label_text_df[label_text_df['school_id']>86]
    label_text_df['bg'] = label_text_df['bg'].map(lambda x: time2unix(x.strip()))
    label_text_df['ed'] = label_text_df['ed'].map(lambda x: time2unix(x.strip()))
    label_text_df = label_text_df.reset_index(drop=True)
    time_step = 45  # 按多少秒分一段
    min_length = 2  # 最短的分词数
    sw = sw_path # 停词list
    label_text_df['subtitle_part'] = ""
    label_text_df['subtitle_cut'] = ""

    # data augment
    df = label_text_df.copy()
    for index, row in label_text_df.iterrows():
        if row['duration'] > 100:  # duration 秒级
            start = row['bg']  # 片段的开始时间
            end = row['ed']  # 片段的结束时间
            for p in range((row['duration'] // time_step)):
                row['duration'] = time_step
                row['bg'] = start + p * time_step * 1000
                row['ed'] = row['bg'] + time_step * 1000
                df = df.append(row, ignore_index=True)
            row['bg'] = row['ed']
            row['ed'] = end
            row['duration'] = (row['ed'] - row['bg']) / 1000
            df = df.append(row, ignore_index=True)
            df.drop(index=index, axis=0, inplace=True)

    label_text_df = df



    data, data_cut, label = [], [], []
    for index, row in label_text_df.iterrows():

        begin_time = row['bg']  # 片段的开始时间
        end_time = row['ed']  # 片段的结束时间

        # 字幕 string2json
        subtitle = row["xf_result"]  # string
        text_part = ""
        if not subtitle.endswith(']'):
            subtitle = subtitle[:subtitle.rfind('}') + 1] + ']'
        subtitle = json.loads(subtitle)  # 所有字幕组成list，包含bg，ed，onebest，speaker

        for text in subtitle:
            cond1 = (int(text['ed']) <= end_time) and (int(text['bg']) >= begin_time)
            cond2 = (int(text['bg']) <= end_time) and (int(text['ed']) >= end_time)
            cond3 = (int(text['ed']) >= begin_time) and (int(text['bg']) <= begin_time)
            if cond1 or cond2 or cond3:
                text_part += text['onebest']  # 字幕
        if text_part != "":
            label_text_df.loc[index, 'subtitle_part'] = text_part  # string类型
            text_cut = word_cut_ch(text_part, sw)
            if len(text_cut.split(" ")) > min_length:
                label_text_df.loc[index, 'subtitle_cut'] = text_cut
    # 把整个文件的字幕文件删掉，只留片段字幕
    label_text_df.drop('xf_result',axis=1,inplace=True)
    # 去掉不确定和没有字幕的片段
    label_text_df = label_text_df[(label_text_df['model_check'] != 0) & (label_text_df['subtitle_part'] != "")]
    print('数据有哪些列：')
    print(label_text_df.columns.values)
    print("最终数据长度：")
    print(label_text_df.shape)
    print("扩充后的数据类型分布：")
    print(label_text_df['type'].value_counts())
    # 只有闲聊的dataframe且part片段要有内容
    label_text_df_type3 = label_text_df[label_text_df['type'] == type]
    # 闲聊的正负例差值，2是正例，1是负例
    model_check = label_text_df_type3['model_check'].value_counts()
    print(model_check)
    diff = model_check[2] - model_check[1]
    print("正负例差别数为： "+str(diff))

    # 将正负例样本变平衡
    index_list = label_text_df_type3.index.values.tolist()
    print(len(index_list))
    # index_list = index_list + random.sample(label_text_df[(label_text_df['subtitle_part'] != "") & (
    #         label_text_df['type'] != type) & (label_text_df['model_check'] == 2)].index.values.tolist(), diff//2)
    for index, row in label_text_df.iterrows():
        if index in index_list:
            text_cut = row['subtitle_cut'].split(" ")  # list
            if len(text_cut) >= min_length:
                # 正例
                if row['type'] == type and row['model_check'] == 2:
                    data.append(row['subtitle_part'])
                    data_cut.append(row['subtitle_cut'])
                    label.append(1)

                # 负例
                elif row['model_check'] == 1 and row['type'] == type:
                    data.append(row['subtitle_part'])
                    data_cut.append(row['subtitle_cut'])

                    label.append(0)

                elif row['type'] != type:
                    data.append(row['subtitle_part'])
                    data_cut.append(row['subtitle_cut'])

                    label.append(0)
    print(len(label))
    print('1:' + str(label.count(1)) + ' 0:' + str(label.count(0)))


    data_length =  np.array(list(map(len, [t.strip().split(' ') for t in data_cut])))
    max_seq_len = sorted(data_length)[-1]
    print(sorted(data_length)[-2])
    # bins = np.linspace(min(data_length), max(data_length), 20)
    # plt.hist(data_length,bins)
    # plt.show()

    data = np.array(data)
    label = np.array(label)
    data_cut = np.array(data_cut)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(len(data)))
        data = data[shuffle_indices]
        label = label[shuffle_indices]
        data_cut = data_cut[shuffle_indices]

    return data, data_cut,label,max_seq_len


def batch_iter(data, label, batch_size, num_epochs):
    """
    A mini-batch iterator to generate mini-batches for training neural network
    :param data: a list of sentences. each sentence is a vector of integers
    :param label: a list of labels
    :param batch_size: the size of mini-batch
    :param num_epochs: number of epochs
    :return: a mini-batch iterator
    """

    assert len(data) == len(label)
    # data = np.array(data)
    # data_size = data.shape[0]
    data_size =len(data)
    epoch_length = data_size // batch_size
    if epoch_length==0:
        epoch_length=1
    for _ in range(num_epochs):
        for batch_num in range(epoch_length):
            start_index = batch_num * batch_size
            end_index = start_index + batch_size

            xdata = data[start_index: end_index]
            ydata = label[start_index: end_index]

            yield xdata, ydata


if __name__ == '__main__':
    path = "../subtitle.xlsx"
    label_df = pd.read_excel(path)
    # print(label_df.describe())
    # label_text_df = add_text(label_df)
    # print(label_text_df['type'].value_counts())
    # print(label_text_df['duration'].value_counts())
    # bins = np.linspace(min(label_text_df['duration']), 500, 20)
    # plt.hist(label_text_df['duration'],bins)
    # plt.show()
    # print(label_text_df.loc[label_text_df['type']=='3','model_check'].value_counts())
    # print(label_text_df.describe())
    data, data_cut, label, max_seq_len = get_data(label_df, sw_path='../stopwords/ch_stopwords.txt',type='2')
    # with open("./data.txt",'w') as f:
    #     for i in data:
    #         f.write(str(i)+"\n")
    train_data = batch_iter(data, label, 32, 50)
    count=0
    for i in train_data:
        count+=1
    print(f"有{count}个batch")
    train = {}
    train['text'] = data
    train['text_cut'] = data_cut
    train['label'] = label

    train = pd.DataFrame(train, columns=['text', 'text_cut', 'label'])
    print("label的正负数量")
    print(train['label'].value_counts())

    # bins = np.linspace(min(data_length), max(data_length), 20)
    # plt.hist(data_length,bins)
    # plt.show()
    # print(max_seq_len)
    # for i in data:
    #     print(len(i.split(" ")),i)
