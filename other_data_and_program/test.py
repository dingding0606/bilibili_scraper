import ssl
ssl._create_default_https_context=ssl._create_unverified_context

import requests
import time
import random
import re
import json
import pandas as pd

from requests_html import HTMLSession

from tqdm import tqdm

# import asyncio
# if asyncio.get_event_loop().is_running(): # Only patch if needed (i.e. running in Notebook, Spyder, etc)
#     import nest_asyncio
#     nest_asyncio.apply()



# 之前使用非官方数据，可以获取历史数据，但由于数据不全最终舍弃了该方法
# url = "http://bz.feigua.cn/ranking/DailyHotVideo/"
# date_id = 20220118
# channel_list = ["155","160","1","3","129","4","36","188","202","119","5","181","167"]

# 需要注意的是在requests当中加入proxy的方法与urllib3有所不同
url = "https://www.bilibili.com/v/"

channel_id = ["douga","music","dance","game","knowledge","tech","sports","car","life","food","animal","kichiku","fashion","ent"]

User_agent=[
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
    "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
    "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
    "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
    "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
    "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
    "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
    "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",
]

proxy = [
    {
    'HTTP':'202.108.22.5:80',
    'HTTPS':'202.108.22.5:80'
    },
    {
    'HTTP':'60.255.151.82:80',
    'HTTPS':'60.255.151.82:80'
    },
    {
    'HTTP':'60.255.151.81:80',
    'HTTPS':'60.255.151.81:80'
    },
    {
    'HTTP':'58.240.52.114:80',
    'HTTPS':'58.240.52.114:80'
    },
    {
    'HTTP':'202.108.22.5:80',
    'HTTPS':'202.108.22.5:80'
    },
]


def request_bilibili_rank(channel_id):
    '''
    :param channel_id: id of channel
    :return: html of website
    '''
    headers = {"User-Agent": random.choice(User_agent)}
    req = requests.get(url+channel_id, headers=headers, proxies=random.choice(proxy))
    html = req.text
    return html


def get_keywords():
    keywords = []
    '''
    This methods scapes all the popular keywords on Bilibili
    '''
    headers = {"User-Agent": random.choice(User_agent)}

    for id in channel_id:


        session = HTMLSession()
        r = session.get(url+id, headers=headers, proxies=random.choice(proxy))

        r.html.render(timeout=1000)
        print(r.text)


#         #req = requests.get(url+id, headers=headers, proxies=random.choice(proxy))
#         html = r.text

#         # channel sub items
#         sub_item = []
#         print(html)

#         regex = "<button class=\"channel-nav-sub-item\">(.*?)<\/button>"
#         pattern = re.compile(regex,re.S)
#         matches = pattern.findall(html)

#         print(matches)

    return keywords


#仿照Youtube视频信息数据集确立bilibili数据集的基本信息项
data_dict = {
    "video_bv":[],
    "trending_data":[],
    "title":[],
    "channel_title":[],
    "publish_time":[],
    "authors":[],
    "views":[],
    "danmaku":[],
    "likes":[],
    "coins":[],
    "shares":[],
    "tags":[],
    "comment_count":[],
    "description":[],
    "cover_links":[],
    'tid':[],
    'duration':[],
    'favorite':[],
    'scores':[],
    "pop":[]
}

video_bv = [] #视频的bv号，在数据库中可以作为key值
tid = [] #分区id
trending_data = [] #进入排行榜的时间，由于官方的排行榜没有给出历史数据，所以这个时间就是对网页进行爬取的时间
title = [] #视频标题
channel_title = [] #分区名称
publish_time = [] #视频的发布时间
authors = [] #视频作者
views = [] #浏览量
danmaku = [] #弹幕数量
likes = [] #点赞数量
coins = [] #硬币数量
shares = [] #分享数量
tags = [] #视频标签
comment_count = [] #评论数量
description = [] #视频描述
cover_links = [] #视频封面链接
favorite = [] #收藏数量
duration = [] #视频时长
scores = [] #视频排行榜得分
pop = [] #是否进入排行榜

get_keywords()

# for j in range(len(channel_id)):
#     html = request_bilibili_rank(channel_id[j]).replace(" ","")
#     #每个视频信息在排行榜中是以json形式的文本展示的
#     rankList = r'"rankList":(.*?),"rankNote":'
#     pattern = re.compile(rankList,re.S)
#     rankList = pattern.findall(html)
#     rankList = rankList[0]
#     #在python中false要改写为False，null要改写成None
#     rankList = rankList.replace('false','False')
#     rankList = rankList.replace('null','None')
#     rankList = eval(rankList)
#     print((j+1)/len(channel_id)*100,"%")
#     print('-'*30)

#     for i in tqdm(rankList):
#         rank_dict = i
#         video_bv.append(rank_dict['bvid'])
#         tid.append(rank_dict['tid'])
# #     stadardTime = time.strftime("%Y-%m-%d %H:%M:%S", rank_dict['ctime'])
#         publish_time.append(rank_dict['ctime'])
#         title.append(rank_dict['title'])
#         channel_title.append(rank_dict['tname'])
#         trending_data.append('2022-02-04')
#         authors.append(rank_dict['owner']['name'])
#         views.append(rank_dict['stat']['view'])
#         danmaku.append(rank_dict['stat']['danmaku'])
#         favorite.append(rank_dict['stat']['favorite'])
#         likes.append(rank_dict['stat']['like'])
#         coins.append(rank_dict['stat']['coin'])
#         scores.append(rank_dict['score'])
#         shares.append(rank_dict['stat']['share'])
#         comment_count.append(rank_dict['stat']['reply'])
#         cover_links.append(rank_dict['pic'])
#         duration.append(rank_dict['duration'])
#         pop.append(1)
