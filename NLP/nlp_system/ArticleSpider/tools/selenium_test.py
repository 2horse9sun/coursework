from selenium import webdriver
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
chrome_options = Options()
chrome_options.add_argument("--disable-extensions")
chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
browser = webdriver.Chrome(executable_path="D:/Documents/Downloads/chromedriver_win32(4)/chromedriver.exe", chrome_options=chrome_options)
browser.get("https://www.taobao.com")