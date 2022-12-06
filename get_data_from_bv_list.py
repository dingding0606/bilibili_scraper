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

    #try:
    bili = BiliSpider()
    sleep(random.randint(1,5))
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
    # except:
    #     print("An error occured in the main part of collect_video_stat() \n")

    try:
        stat['get_time'] = time.time() #china_time_now().strftime("%m/%d/%Y, %H:%M:%S")
    except:
        print("An error occured in the ----get_time---- part of collect_video_stat() \n")

    with output_lock:
        with open("new_final_output_day1.txt", "a") as file:
            file.write(json.dumps(stat, indent=4))
            file.write("\n\n\n\n\n")

        with open("new_final_output_day1_status.txt", "a") as file:
            file.write(bv)
            file.write("  " + str(len(str(stat))))
            file.write("\n")

    return 1


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


def progress_indicator(future):
    global lock, tasks_total, tasks_completed
    # obtain the lock
    with lock:
        # update the counter
        tasks_completed += 1
        # report progress
        print(f'{tasks_completed}/{tasks_total} completed, {tasks_total-tasks_completed} remain.')



def main():

    # read in BV lists and remove repeatitions
    file1 = open('NEW_FINAL_BV_LIST.txt', 'r')
    lines = file1.readlines()

    bv_list = set()

    for line in lines:
        bv_list.update([line[:-1]])

    bv_list = list(bv_list)

    print("Total # of bv after removing duplicates: " + str(len(bv_list)))
    print("BEGIN WRITING INTO FILES ... \n")

    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(collect_video_stat, bv) for bv in bv_list]

        for future in futures:
            future.add_done_callback(progress_indicator)

    # for bv in bv_list:
    #     collect_video_stat(bv)

    # # 'a' = append mode
    # with open("Nov_24_output.txt", "a") as file:
    #     for result in results:
    #         file.write(json.dumps(result, indent=4))
    #         file.write("\n\n\n\n\n")

    print("DONE WITH WRITING INTO FILES :) \n")


output_lock = Lock()

# For progress indicator
lock = Lock()
tasks_total = 12912
tasks_completed = 0

main()
