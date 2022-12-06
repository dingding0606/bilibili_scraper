from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import re

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

driver.maximize_window()

# <button class="channel-nav-sub-item">影视杂谈</button>
# class="name">影视杂谈</a>
#elements = driver.find_element_by_class_name("channel-nav-sub-item")

driver.get("https://www.bilibili.com/v/cinephile")

html = driver.page_source

#print(html)

regex = "class=\"name\">(.*?)</a>"
pattern = re.compile(regex, re.S)
keywords = pattern.findall(html)

final_list = []

for keyword in keywords:
    if "span" not in keyword:
        final_list.append(keyword)

print(final_list)

driver.quit()
