import csv
import json

def compress_tag_info(tags_stat):
    num_of_tags = len(tags_stat)
    total_tag_subscriber_count = 0
    total_tag_archive_count = 0

    for tag in tags_stat:
        total_tag_subscriber_count += int(tag['tag_subscriber_count'])

        this_tag_archieve_count = 0

        if '\u4e07' in tag['tag_archive_count']:
            count = tag['tag_archive_count']
            this_tag_archieve_count = float(count[0]) * 10000

        total_tag_archive_count += this_tag_archieve_count

    result = {}
    result['num_of_tags'] = num_of_tags
    result['total_tag_subscriber_count'] = total_tag_subscriber_count
    result['total_tag_archive_count'] = total_tag_archive_count

    return result


FROM_FILE_NAME = "DATA_ONE_DAY.txt"
TO_FILE_NAME = "DATA_ONE_DAY_CSV.csv"

from_file = open(FROM_FILE_NAME, "r")

feature_names = ['bv', 'copyright', 'pubdate', 'dim_width', 'dim_height', 'dim_rotate',
                    'duration', 'is_story', 'cover_pic', 'uploader_id', 'is_cooperation',
                    'is_movie', 'no_reprint', 'no_share', 'hd5', 'is_360', 'is_stein_gate',
                    'pgc_pay', 'ugc_pay', 'view', 'coin', 'danmu', 'favorite', 'like',
                    'share', 'reply', 'his_rank', 'now_rank', 'argue_msg', 'tid', 'tname',
                    'title', 'videos', 'num_of_tags', 'total_tag_subscriber_count',
                    'total_tag_archive_count', 'up_archive_count', 'up_article_count',
                    'up_follower', 'up_like_num', 'up_name', 'up_sex', 'up_signature',
                    'up_following', 'up_avatar', 'up_is_official', 'up_level', 'get_time']

special_features = ['dim_width', 'dim_height', 'dim_rotate']

data = from_file.read()
video_info_list = data.split("\n\n\n\n\n")
video_info_list.pop() # drop the last element (empty)



# WRITE INTO CSV
with open(TO_FILE_NAME, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(feature_names)

    for video_info in video_info_list:
        video_info = json.loads(video_info)

        video_info['tags_stat'] = compress_tag_info(video_info['tags_stat'])

        row = []
        for feature in feature_names:
            if "dim_" in feature:
                feature = feature.split("_")[1]
                row.append(video_info['dimension'][feature])
            elif "_tag" in feature:
                row.append(video_info['tags_stat'][feature])
            else:
                row.append(video_info[feature])

        writer.writerow(row)
