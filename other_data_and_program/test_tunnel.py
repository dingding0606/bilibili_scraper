
import requests

# 隧道域名:端口号
tunnel = "q848.kdltps.com:15818"

# 用户名密码方式
username = "t16957939048141"
password = "hxmx4w5v"
proxies = {
    "http": "http://%(user)s:%(pwd)s@%(proxy)s/" % {"user": username, "pwd": password, "proxy": tunnel},
    "https": "http://%(user)s:%(pwd)s@%(proxy)s/" % {"user": username, "pwd": password, "proxy": tunnel}
}

# 白名单方式（需提前设置白名单）
# proxies = {
#     "http": "http://%(proxy)s/" % {"proxy": tunnel},
#     "https": "http://%(proxy)s/" % {"proxy": tunnel}
# }

# 要访问的目标网页
keyword = "ipad"
target_url =  "https://search.bilibili.com/all?vt=82687596&keyword=LOL&order=pubdate"

# 使用隧道域名发送请求
response = requests.get(target_url, proxies=proxies)

# 获取页面内容
if response.status_code == 200:
    print(response.text)  # 请勿使用keep-alive复用连接(会导致隧道不能切换IP)
