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
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from datetime import datetime
from dateutil import tz
from BiliSpider import BiliSpider
from threading import Lock
from time import sleep

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
    {
    'HTTP':'117.251.103.186:8080',
    'HTTPS':'117.251.103.186:8080'
    },
    {
    'HTTP':'208.82.61.38:3128',
    'HTTPS':'208.82.61.38:3128'
    },
    {
    'HTTP':'64.29.86.251:3129',
    'HTTPS':'64.29.86.251:3129'
    },
    {
    'HTTP':'155.4.244.218:80',
    'HTTPS':'155.4.244.218:80'
    },
    {
    'HTTP':'208.82.61.13:3128',
    'HTTPS':'208.82.61.13:3128'
    },
]
# #
# proxy = [
#     {
#     'http':'http://198.27.74.6:9300',
#     'https':'http://198.27.74.6:9300',
#     },
# ]



def get_lastest_videos(keyword):
    '''
    Given a String keyword, search on bilibili using this keyword,
    and return a list of video ids (BV number) such that the video is
    published within 1 hour.
    '''

    sleep(random.randint(1,5))

    videos = []

    search_url = "https://search.bilibili.com/all?keyword=" + keyword + "&order=pubdate" + "&page="
    bv_regex = "<a href=\"//www.bilibili.com/video/(.*?)\?from=search"
    # headers = {"User-Agent": random.choice(User_agent)}

    page_num = 1
    MAX_PAGE_NUM = 5

    within_one_hour = True

    # Page by Page, quit the loop while finding something beyond 24 hours
    while(page_num <= MAX_PAGE_NUM and within_one_hour):

        page_trail = 0
        PAGE_MAX_TRIALS = 5
        bv_list = []

        while page_trail < PAGE_MAX_TRIALS and len(bv_list) == 0:
            try:
                page_trail += 1
                headers = {"User-Agent": random.choice(User_agent)}
                search_page = requests.get(search_url + str(page_num), headers=headers, proxies=random.choice(proxy), timeout=15)
                search_page_html = search_page.text # html for the current search page
                pattern = re.compile(bv_regex, re.S)
                bv_list = pattern.findall(search_page_html) # bv_list contains all the BV numbers (video ids) in the current page

            except Exception as e:
                print("Oops....!", e.__class__, "occurred.")
                sleep(random.randint(5,10) * page_trail)

        print("Length of list: " + str(len(bv_list)))

        # From the BVs in this page, select BVs that are within start_time and end_time
        for bv in bv_list:
            sleep(random.randint(1,5))

            trail = 0
            MAX_TRIALS = 5

            while trail < MAX_TRIALS:
                headers = {"User-Agent": random.choice(User_agent)}
                try:
                    trail += 1

                    video_page_url = "https://www.bilibili.com/video/" + str(bv)
                    video_page = requests.get(video_page_url, proxies=random.choice(proxy), headers=headers, timeout=15)
                    video_page_html = video_page.text
                    publish_time_from_now = time_from_now(publish_time(video_page_html))

                    if (to_hours(publish_time_from_now) > 0):
                        within_one_hour = False
                    else:
                        videos.append(bv)

                    break
                except Exception as e:
                    print("Oops!", e.__class__, "occurred.")
                    sleep(random.randint(3,6) * trail)

        page_num += 1

    # directly write into this file...
    with bv_file_lock:
        with open("BV_list_using_tunnel.txt", "a") as file:
            for bv in videos:
                file.write(bv + "\n")

    # a list of bv's
    print(videos)
    return videos


def to_hours(duration):
    '''
    Convert duration to how many hours
    '''
    duration_in_s = duration.total_seconds()
    hours = divmod(duration_in_s, 3600)[0]
    return int(hours)


def publish_time(video_page_html):
    '''
    Given the html text of the video page, return the publish time of this video
    in China time zone.
    '''
    regex = "<span class=\"pudate-text\">\n +(.*?)\n +</span>"
    pattern = re.compile(regex, re.S)
    time = pattern.findall(video_page_html)
    standardized_time = datetime.strptime(time[0], "%Y-%m-%d %H:%M:%S")

    # Mark time zone info as "Asia/Shanghai"
    standardized_time = standardized_time.replace(tzinfo=tz.gettz('Asia/Shanghai'))

    # This time object is marked as China timezone
    return standardized_time


def china_time_now():
    now = datetime.utcnow()
    now = now.replace(tzinfo=tz.gettz('UTC'))
    china_time_now = now.astimezone(tz.gettz('Asia/Shanghai'))
    return china_time_now


def time_from_now(china_time_past):
    '''
    Given a time object in china timezone, return the time from now to
    that time.
    '''
    return china_time_now() - china_time_past


def collect_video_stat(bv):
    '''
    Given a video id (BV number), return a dictionary containing stats related to this video
    (including get_time (time stamp of current time))

    bv, uploader_id, pubtime, view_count, like_count, star_count, share_count, danmu_count,
    comment_count, title (text), description (text), tags (text), tag_subscriber_count,
    comment_like_count, video_len, top_5_comments (text), get_time
    '''

    stat = {}

    stat["bv"] = bv

    try:
        bili = BiliSpider()
        video_json = bili.get_video_stat(bv)
        data = video_json['data']

        view = data['View'] # basic info about this video
        tags = data['Tags'] # information about the tags of this video
        card = data['Card'] # information about uploader

        # meaning of these keywords are here: https://github.com/SocialSisterYi/bilibili-API-collect/blob/9c467bbebc0f28b5fe7372401b3203475e3f4d19/video/info.md
        stat['copyright'] = view['copyright']
        stat['ctime'] = view['ctime']
        stat['pubdate'] = view['pubdate']
        stat['dimension'] = view['dimension']
        stat['duration'] = view['duration']
        stat['is_story'] = view['is_story']
        stat['cover_pic'] = view['pic']
        stat['uploader_id'] = view['owner']['mid']

        stat['is_cooperation'] = view['rights']['is_cooperation']
        stat['is_movie'] = view['rights']['movie']
        stat['no_reprint'] = view['rights']['no_reprint']
        stat['no_share'] = view['rights']['no_share']
        stat['hd5'] = view['rights']['hd5']
        stat['is_360'] = view['rights']['is_360']
        stat['is_stein_gate'] = view['rights']['is_stein_gate']
        stat['pgc_pay'] = view['rights']['pay']
        stat['ugc_pay'] = view['rights']['ugc_pay']

        stat['view'] = view['stat']['view']
        stat['coin'] = view['stat']['coin']
        stat['danmu'] = view['stat']['danmaku']
        stat['favorite'] = view['stat']['favorite']
        stat['like'] = view['stat']['like']
        stat['share'] = view['stat']['share']
        stat['reply'] = view['stat']['reply']
        stat['his_rank'] = view['stat']['his_rank']
        stat['now_rank'] = view['stat']['now_rank']
        stat['argue_msg'] = view['stat']['argue_msg']

        stat['tid'] = view['tid']
        stat['tname'] = view['tname']
        stat['title'] = view['title']
        stat['videos'] = view['videos'] # total number of videos in this collection

        stat['tags_stat'] = collect_tag_stat(tags)

        stat.update(collect_uploader_stat(card))
    except:
        print("An error occured in the main part of collect_video_stat() \n")

    try:
        stat['get_time'] = china_time_now().strftime("%m/%d/%Y, %H:%M:%S")
    except:
        print("An error occured in the ----get_time---- part of collect_video_stat() \n")

    return stat


def collect_tag_stat(tags):
    '''
    tags: a list of dicts, each dict contains info of a tag
    return: a list of tag stats.
    '''
    tags_stat = []

    for tag in tags:
        useful_info = {}

        useful_info['tag_id'] = tag['tag_id']
        useful_info['tag_name'] = tag['tag_name']
        useful_info['tag_subscriber_count'] = tag['subscribed_count']
        useful_info['tag_featured_count'] = tag['featured_count']
        useful_info['tag_archive_count'] = tag['archive_count']

        tags_stat.append(useful_info)

    return tags_stat


def collect_uploader_stat(card):
    '''
    Given an uploader's card, return a dictionary containing stats related to this uploader

    p.s. card is a dictionary containing the uploader's info

    Description of features: https://github.com/SocialSisterYi/bilibili-API-collect/blob/master/user/info.md
    '''

    uploader_stat = {}

    uploader_stat['up_archive_count'] = card['archive_count']
    uploader_stat['up_article_count'] = card['article_count']
    uploader_stat['up_follower'] = card['follower'] # UP主粉丝数
    uploader_stat['up_like_num'] = card['like_num'] # UP主获赞次数

    uploader_stat['up_name'] = card['card']['name']
    uploader_stat['up_sex'] = card['card']['sex']
    uploader_stat['up_signature'] = card['card']['sign']
    uploader_stat['up_following'] = card['card']['attention'] # UP主关注数 following
    uploader_stat['up_avatar'] = card['card']['face']
    uploader_stat['up_is_official'] = card['card']['Official']['type'] # -1 无认证，0 认证
    uploader_stat['up_level'] = card['card']['level_info']['current_level']

    return uploader_stat


def simple_progress_indicator(result):
    print('.', end='', flush=True)


def progress_indicator(future):
    global progress_lock, tasks_total, tasks_completed
    # obtain the lock
    with progress_lock:
        # update the counter
        tasks_completed += 1
        # report progress
        print(f'{tasks_completed}/{tasks_total} completed, {tasks_total-tasks_completed} remain.')



def main():
    # Get keywords to search for from file
    KEYWORD_FILE = 'tag_names.txt'

    keywords = []

    with open(KEYWORD_FILE) as file:
        for line in file:
            keywords.append(line.rstrip())

    must_include_keywords = ['MAD', 'AMV', 'MMD', '3D', '短片', '手书', '配音', '手办', '模玩', '特摄', '动漫杂谈',
                '综合', '连载动画', '完结动画', '资讯', '官方延伸', '新番时间表', '番剧索引',
                '原创音乐', '翻唱', '演奏', 'VOCALOID', 'UTAU', '音乐现场', 'MV', '乐评盘点',
                '音乐教学', '音乐综合', '说唱', '国产动画', '国产原创相关', '布袋戏',
                '动态漫', '广播剧', '国产动画索引', '宅舞', '街舞',
                '明星舞蹈', '中国舞', '舞蹈综合', '舞蹈教程', '单机游戏', '电子竞技',
                '手机游戏', '网络游戏', '桌游棋牌', 'GMV', '音游', 'Mugen', '游戏赛事',
                '科学科普', '社科', '法律', '心理', '人文历史', '财经商业', '校园学习', '职业职场',
                '设计·创意', '野生技能协会', '数码', '软件应用', '计算机技术', '科工机械',
                '极客DIY', '搞笑', '亲子', '出行', '三农', '家居房产', '手工', '绘画',
                '日常', '鬼畜调教', '音MAD', '人力VOCALOID', '鬼畜剧场', '教程演示',
                '美妆护肤', '仿妆cos', '穿搭', '时尚潮流', '热点', '环球', '社会',
                '综合', '综艺', '娱乐杂谈', '粉丝创作', '明星综合', '影视杂谈',
                '影视剪辑', '小剧场', '预告·资讯', '纪录片', '电影', '电视剧',
                '虚拟UP主', '搞笑', '美食', '动物圈', '单机游戏', '运动', '汽车',
                'VLOG', '全部', '网游', '手游', '单机', '娱乐', '电台', '虚拟',
                '生活', '学习', '影视杂谈', '影视剪辑', '小剧场', '预告']

    keywords = random.sample(keywords, 500)

    keywords = must_include_keywords + keywords

    print("Total number of keywords: " + str(len(keywords)))

    # Get videos published within the past 1 hour from the search results using keywords
    print("BEGIN SCRAPING ... \n")

    # for keyword in keywords:
    #     get_lastest_videos(keyword)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(get_lastest_videos, keyword) for keyword in keywords]

        for future in futures:
            future.add_done_callback(progress_indicator)

    print("DONE WITH SCRAPING ... \n")

    #
    # # Remove Duplicates of BVid
    # bv_list = set()
    #
    # for future in as_completed(futures):
    #     bv_list.update(future.result())
    #
    #
    # # append all the bv numbers into this file
    # with open("BV_list.txt", "a") as file:
    #     for bv in bv_list:
    #         file.write(bv + "\n")
    #
    #
    # total = len(bv_list)
    # print("Total number of videos (without duplicates): " + str(total) + "\n")
    #
    #
    # # Collect video stats for each BVid and write into a file
    # print("BEGIN WRITING INTO FILES ... \n")
    #
    # with ThreadPoolExecutor(max_workers=100) as p:
    #     results = p.map(collect_video_stat, bv_list)
    #
    # # 'a' = append mode
    # with open("dc_output.txt", "a") as file:
    #     for result in results:
    #         file.write(json.dumps(result, indent=4))
    #         file.write("\n\n\n\n\n")
    #
    # print("DONE WITH WRITING INTO FILES :) \n")

progress_lock = Lock()
bv_file_lock = Lock()
tasks_total = 5153
tasks_completed = 0

main()
