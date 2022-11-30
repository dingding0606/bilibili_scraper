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

    stat['desc'] = view['desc']

    with output_lock:
        with open("VIDEO_DESCRIPTION.txt", "a") as file:
            file.write(json.dumps(stat, indent=4))
            file.write("\n\n\n\n\n")

    return 1


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
    file1 = open('FINAL_BV_LIST.txt', 'r')
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

    print("DONE WITH WRITING INTO FILES :) \n")


output_lock = Lock()

# For progress indicator
lock = Lock()
tasks_total = 10000
tasks_completed = 0

main()
