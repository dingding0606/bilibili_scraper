#!/usr/bin/env python
# _*_ coding:utf-8 _*_
#
# @Version : 1.0
# @Time    : 2018/10/25
# @Author  : 圈圈烃
# @File    : BiliSpider
# @Description:
#
#
import requests
import json
import random


class BiliSpider:
    def __init__(self):

        self.online_api = "https://api.bilibili.com/x/web-interface/online"  # 在线人数
        self.video_api = "https://api.bilibili.com/x/web-interface/archive/stat?&bvid=%s"    # 视频信息
        self.newlist_api = "https://api.bilibili.com/x/web-interface/newlist?&rid=%s&pn=%s&ps=%s"     # 最新视频信息
        self.region_api = "https://api.bilibili.com/x/web-interface/dynamic/region?&rid=%s&pn=%s&ps=%s"  # 最新动态信息
        self.member_api = "http://space.bilibili.com/ajax/member/GetInfo"  # 用户信息
        self.stat_api = "https://api.bilibili.com/x/relation/stat?vmid=%s"  # 用户关注数和粉丝总数
        self.upstat_api = "https://api.bilibili.com/x/space/upstat?mid=%s"     # 用户总播放量和总阅读量
        self.follower_api = "https://api.bilibili.com/x/relation/followings?vmid=%s&pn=%s&ps=%s"    # 用户关注信息
        self.fans_api = "https://api.bilibili.com/x/relation/followers?vmid=%s&pn=%s&ps=%s"    # 用户粉丝信息

    def get_api(api_url):

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

        # headers = {
        #     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        #     "Accept-Encoding": "gzip, deflate, br",
        #     "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
        #     "Host": "api.bilibili.com",
        #     "Referer": "https://www.bilibili.com/",
        #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36",
        # }

        headers = {"User-Agent": random.choice(User_agent)}
        res = requests.get(api_url, headers=headers, proxies=random.choice(proxy))
        res_dict = res.json()
        return res_dict

    def get_online(self):
        """
        获取在线信息
        all_count: 最新投稿
        web_online: 在线人数
        :return:
        """
        online_dic = BiliSpider.get_api(self.online_api)
        return online_dic

    def get_video_info(self, bvid):
        """
        获取视频信息
        :param aid: 视频id
        :return:

        {'code': 0, 'message': '0', 'ttl': 1, 'data': {'aid': 732803611, 'bvid': 'BV18D4y1s7mj', 'view': 62, 'danmaku': 0, 'reply': 1, 'favorite': 1, 'coin': 0, 'share': 0, 'like': 2, 'now_rank': 0, 'his_rank': 0, 'no_reprint': 1, 'copyright': 1, 'argue_msg': '', 'evaluation': ''}}


        """
        res = BiliSpider.get_api(self.video_api %bvid)
        # print(res)
        return res

    def get_newlist_info(self, rid, pn, ps):
        """
        获取最新视频信息
        :param rid: 二级标题的id (详见tid_info.txt)
        :param pn:  页数
        :param ps:  每页条目数 1-50
        :return:
        """
        res = BiliSpider.get_api(self.newlist_api %(rid, pn, ps))
        return res

    def get_region_info(self, rid, pn, ps):
        """
        获取最新视频信息
        :param rid: 二级标题的id (详见tid_info.txt)
        :param pn:  页数
        :param ps:  每页条目数 1-50
        :return:
        """
        res = BiliSpider.get_api(self.region_api %(rid, pn, ps))
        return res

    def get_member_info(self, mid):
        """
        获取用户信息
        :param mid:用户id
        :return:
        """
        post_data = {
            "crsf": "",
            "mid": mid,
        }
        header = {
            "Host": "space.bilibili.com",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:63.0) Gecko/20100101 Firefox/63.0",
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
            "Referer": "https://www.bilibili.com/",
        }
        res = requests.post(self.member_api, data=post_data, headers=header)
        member_dic = json.dumps(res.json(), ensure_ascii=False)
        return member_dic

    def get_stat_info(self, vmid):
        """
        获取某用户的关注数和粉丝总数
        :param vmid: 用户id
        :return:
        """
        res = BiliSpider.get_api(self.stat_api % vmid)
        return res

    def get_upstat_info(self, mid):
        """
        获取某用户的总播放量和总阅读量
        :param mid: 用户id
        :return:
        """
        res = BiliSpider.get_api(self.upstat_api % mid)
        return res

    def get_follower_info(self, vmid, pn, ps):
        """
        获取某用户关注者信息
        :param vmid:用户id
        :param pn:  页数 最多5页
        :param ps:  每页条目数 1-50
        :return:
        """
        res = BiliSpider.get_api(self.follower_api %(vmid, pn, ps))
        return res

    def get_fans_info(self, vmid, pn, ps):
        """
        获取某用户粉丝信息
        :param vmid:用户id
        :param pn:  页数 最多5页
        :param ps:  每页条目数 1-50
        :return:
        """
        res = BiliSpider.get_api(self.fans_api %(vmid, pn, ps))
        return res


if __name__ == '__main__':
    bili = BiliSpider()
    res = bili.get_video_info("BV18D4y1s7mj")
    print(res)
