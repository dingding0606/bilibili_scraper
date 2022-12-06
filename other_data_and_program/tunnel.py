"""快代理代理IP使用教程"""

import requests
import random

# 要访问的目标网页
page_url = "https://www.bilibili.com"

# 隧道的host与端口
proxy = "w486.kdltps.com:15818"

# 用户名和密码(隧道代理分配的)
username = "t16922028456997"
password = "h249on7r"

# 代理IP的格式
proxies = {
     "http": "http://%(user)s:%(pwd)s@%(proxy)s/" % {'user': username, 'pwd': password, 'proxy': proxy},
     "https": "https://%(user)s:%(pwd)s@%(proxy)s/" % {'user': username, 'pwd': password, 'proxy': proxy
     }}

# 添加header，模拟用户请求
headers = {
    "Accept-Encoding": "Gzip",  # 使用gzip压缩传输数据让访问更快
    "User-Agent": "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6"
}

# 发送request请求,打印响应code与body内容
r = requests.get(url=page_url, proxies=proxies, headers=headers, timeout=10, ver)
print("response code",r.status_code)
print("response body",r.text)
