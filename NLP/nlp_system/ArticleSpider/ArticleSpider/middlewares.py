# -*- coding: utf-8 -*-

# Define here the models for your spider middleware
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/spider-middleware.html

from scrapy import signals
from fake_useragent import UserAgent

from tools.crawl_xici_ip import GetIP

proxy_api = {"code":200,"msg":"","data":[{"ip":"117.64.251.33","port":14568,"expire_time":"2020-11-09 20:49:41","city":"\u5408\u80a5","isp":"\u7535\u4fe1"},{"ip":"117.95.106.214","port":7791,"expire_time":"2020-11-09 20:46:37","city":"\u6dee\u5b89","isp":"\u7535\u4fe1"},{"ip":"60.175.9.200","port":62563,"expire_time":"2020-11-09 20:54:50","city":"\u5b89\u5e86","isp":"\u7535\u4fe1"},{"ip":"49.87.250.40","port":940,"expire_time":"2020-11-09 20:43:48","city":"\u6dee\u5b89","isp":"\u7535\u4fe1"},{"ip":"123.134.68.196","port":14568,"expire_time":"2020-11-09 21:02:12","city":"\u83b1\u829c","isp":"\u8054\u901a"},{"ip":"117.91.165.22","port":271,"expire_time":"2020-11-09 21:03:33","city":"\u626c\u5dde","isp":"\u7535\u4fe1"},{"ip":"49.87.254.82","port":271,"expire_time":"2020-11-09 20:56:33","city":"\u6dee\u5b89","isp":"\u7535\u4fe1"},{"ip":"61.179.94.247","port":940,"expire_time":"2020-11-09 20:43:24","city":"\u70df\u53f0","isp":"\u8054\u901a"},{"ip":"58.255.5.54","port":271,"expire_time":"2020-11-09 20:36:39","city":"\u60e0\u5dde","isp":"\u8054\u901a"},{"ip":"117.65.0.192","port":14568,"expire_time":"2020-11-09 20:43:50","city":"\u868c\u57e0","isp":"\u7535\u4fe1"},{"ip":"114.230.126.20","port":14568,"expire_time":"2020-11-09 20:44:24","city":"\u626c\u5dde","isp":"\u7535\u4fe1"},{"ip":"49.86.26.46","port":4945,"expire_time":"2020-11-09 20:44:19","city":"\u626c\u5dde","isp":"\u7535\u4fe1"},{"ip":"114.107.148.216","port":14568,"expire_time":"2020-11-09 20:47:13","city":"\u5bbf\u5dde","isp":"\u7535\u4fe1"},{"ip":"124.113.243.83","port":7791,"expire_time":"2020-11-09 20:42:38","city":"\u5bbf\u5dde","isp":"\u7535\u4fe1"},{"ip":"175.165.230.22","port":14568,"expire_time":"2020-11-09 21:03:27","city":"\u76d8\u9526","isp":"\u8054\u901a"},{"ip":"122.230.156.46","port":271,"expire_time":"2020-11-09 21:01:31","city":"\u6e56\u5dde","isp":"\u7535\u4fe1"},{"ip":"117.69.180.63","port":271,"expire_time":"2020-11-09 20:42:50","city":"\u9ec4\u5c71","isp":"\u7535\u4fe1"},{"ip":"114.238.217.147","port":940,"expire_time":"2020-11-09 20:56:23","city":"\u6dee\u5b89","isp":"\u7535\u4fe1"},{"ip":"36.56.144.246","port":62563,"expire_time":"2020-11-09 20:58:17","city":"\u6c60\u5dde","isp":"\u7535\u4fe1"},{"ip":"58.252.201.18","port":14568,"expire_time":"2020-11-09 20:41:26","city":"\u60e0\u5dde","isp":"\u8054\u901a"},{"ip":"125.87.96.73","port":62563,"expire_time":"2020-11-09 20:44:13","city":"\u91cd\u5e86","isp":"\u7535\u4fe1"},{"ip":"223.241.30.230","port":4945,"expire_time":"2020-11-09 20:42:29","city":"\u4eb3\u5dde","isp":"\u7535\u4fe1"},{"ip":"117.69.152.117","port":7791,"expire_time":"2020-11-09 20:48:07","city":"\u9ec4\u5c71","isp":"\u7535\u4fe1"},{"ip":"49.87.250.13","port":4945,"expire_time":"2020-11-09 21:03:22","city":"\u6dee\u5b89","isp":"\u7535\u4fe1"},{"ip":"49.87.194.187","port":62563,"expire_time":"2020-11-09 20:41:41","city":"\u6dee\u5b89","isp":"\u7535\u4fe1"},{"ip":"114.99.6.193","port":62563,"expire_time":"2020-11-09 20:54:42","city":"\u94dc\u9675","isp":"\u7535\u4fe1"},{"ip":"117.64.251.139","port":62563,"expire_time":"2020-11-09 21:00:36","city":"\u5408\u80a5","isp":"\u7535\u4fe1"},{"ip":"114.107.150.154","port":940,"expire_time":"2020-11-09 20:37:58","city":"\u5bbf\u5dde","isp":"\u7535\u4fe1"},{"ip":"223.215.9.32","port":62563,"expire_time":"2020-11-09 20:41:50","city":"\u9ec4\u5c71","isp":"\u7535\u4fe1"},{"ip":"114.227.163.207","port":4945,"expire_time":"2020-11-09 20:50:38","city":"\u5e38\u5dde","isp":"\u7535\u4fe1"},{"ip":"114.97.124.135","port":14568,"expire_time":"2020-11-09 20:48:12","city":"\u868c\u57e0","isp":"\u7535\u4fe1"},{"ip":"125.123.140.39","port":7791,"expire_time":"2020-11-09 20:52:58","city":"\u5609\u5174","isp":"\u7535\u4fe1"},{"ip":"123.134.116.124","port":14568,"expire_time":"2020-11-09 20:37:17","city":"\u83b1\u829c","isp":"\u8054\u901a"},{"ip":"183.92.236.137","port":940,"expire_time":"2020-11-09 20:47:36","city":"\u968f\u5dde","isp":"\u8054\u901a"},{"ip":"218.91.7.32","port":62563,"expire_time":"2020-11-09 20:36:25","city":"\u626c\u5dde","isp":"\u7535\u4fe1"},{"ip":"114.97.125.41","port":7791,"expire_time":"2020-11-09 20:43:24","city":"\u868c\u57e0","isp":"\u7535\u4fe1"},{"ip":"122.190.54.25","port":271,"expire_time":"2020-11-09 20:57:18","city":"\u968f\u5dde","isp":"\u8054\u901a"},{"ip":"175.173.220.110","port":62563,"expire_time":"2020-11-09 20:43:32","city":"\u76d8\u9526","isp":"\u8054\u901a"},{"ip":"114.96.168.73","port":7791,"expire_time":"2020-11-09 20:37:25","city":"\u6dee\u5317","isp":"\u7535\u4fe1"},{"ip":"183.92.229.182","port":7791,"expire_time":"2020-11-09 20:57:32","city":"\u968f\u5dde","isp":"\u8054\u901a"},{"ip":"117.69.171.128","port":14568,"expire_time":"2020-11-09 20:52:22","city":"\u9ec4\u5c71","isp":"\u7535\u4fe1"},{"ip":"180.104.215.198","port":4945,"expire_time":"2020-11-09 20:46:14","city":"\u5f90\u5dde","isp":"\u7535\u4fe1"},{"ip":"114.97.140.208","port":4945,"expire_time":"2020-11-09 20:55:46","city":"\u5408\u80a5","isp":"\u7535\u4fe1"},{"ip":"175.173.220.27","port":940,"expire_time":"2020-11-09 20:43:28","city":"\u76d8\u9526","isp":"\u8054\u901a"},{"ip":"114.230.122.106","port":271,"expire_time":"2020-11-09 20:58:38","city":"\u626c\u5dde","isp":"\u7535\u4fe1"},{"ip":"223.241.21.86","port":14568,"expire_time":"2020-11-09 20:56:19","city":"\u4eb3\u5dde","isp":"\u7535\u4fe1"},{"ip":"117.67.246.157","port":14568,"expire_time":"2020-11-09 20:54:17","city":"\u5408\u80a5","isp":"\u7535\u4fe1"},{"ip":"61.132.171.47","port":271,"expire_time":"2020-11-09 20:46:02","city":"\u94dc\u9675","isp":"\u7535\u4fe1"},{"ip":"49.82.27.65","port":14568,"expire_time":"2020-11-09 20:46:22","city":"\u6dee\u5b89","isp":"\u7535\u4fe1"},{"ip":"218.95.54.183","port":7791,"expire_time":"2020-11-09 20:38:33","city":"\u840d\u4e61","isp":"\u7535\u4fe1"},{"ip":"122.190.63.16","port":271,"expire_time":"2020-11-09 20:47:14","city":"\u968f\u5dde","isp":"\u8054\u901a"},{"ip":"115.208.84.163","port":14568,"expire_time":"2020-11-09 21:01:27","city":"\u6e56\u5dde","isp":"\u7535\u4fe1"},{"ip":"27.40.92.16","port":4945,"expire_time":"2020-11-09 20:41:13","city":"\u60e0\u5dde","isp":"\u8054\u901a"},{"ip":"221.202.99.142","port":7791,"expire_time":"2020-11-09 21:03:42","city":"\u76d8\u9526","isp":"\u8054\u901a"},{"ip":"114.230.107.51","port":7791,"expire_time":"2020-11-09 20:49:38","city":"\u626c\u5dde","isp":"\u7535\u4fe1"},{"ip":"49.86.58.89","port":271,"expire_time":"2020-11-09 21:01:34","city":"\u626c\u5dde","isp":"\u7535\u4fe1"},{"ip":"114.96.169.254","port":4945,"expire_time":"2020-11-09 20:58:23","city":"\u6dee\u5317","isp":"\u7535\u4fe1"},{"ip":"182.108.166.156","port":940,"expire_time":"2020-11-09 20:39:34","city":"\u8d63\u5dde","isp":"\u7535\u4fe1"},{"ip":"124.113.241.223","port":62563,"expire_time":"2020-11-09 20:38:05","city":"\u5bbf\u5dde","isp":"\u7535\u4fe1"},{"ip":"180.126.211.120","port":271,"expire_time":"2020-11-09 20:41:35","city":"\u76d0\u57ce","isp":"\u7535\u4fe1"},{"ip":"119.86.214.116","port":62563,"expire_time":"2020-11-09 20:51:19","city":"\u91cd\u5e86","isp":"\u7535\u4fe1"},{"ip":"117.44.42.143","port":14568,"expire_time":"2020-11-09 20:39:15","city":"\u8d63\u5dde","isp":"\u7535\u4fe1"},{"ip":"117.64.148.94","port":271,"expire_time":"2020-11-09 20:57:41","city":"\u5408\u80a5","isp":"\u7535\u4fe1"},{"ip":"119.86.214.86","port":271,"expire_time":"2020-11-09 21:01:28","city":"\u91cd\u5e86","isp":"\u7535\u4fe1"},{"ip":"218.73.117.23","port":14568,"expire_time":"2020-11-09 20:57:20","city":"\u5609\u5174","isp":"\u7535\u4fe1"},{"ip":"27.204.20.0","port":7791,"expire_time":"2020-11-09 21:02:33","city":"\u83b1\u829c","isp":"\u8054\u901a"},{"ip":"119.133.17.152","port":4945,"expire_time":"2020-11-09 20:49:31","city":"\u6c5f\u95e8","isp":"\u7535\u4fe1"},{"ip":"117.64.250.213","port":271,"expire_time":"2020-11-09 20:37:37","city":"\u5408\u80a5","isp":"\u7535\u4fe1"},{"ip":"175.146.71.106","port":62563,"expire_time":"2020-11-09 20:58:18","city":"\u978d\u5c71","isp":"\u8054\u901a"},{"ip":"117.87.208.84","port":271,"expire_time":"2020-11-09 20:56:43","city":"\u5f90\u5dde","isp":"\u7535\u4fe1"},{"ip":"121.230.84.18","port":62563,"expire_time":"2020-11-09 20:37:46","city":"\u6cf0\u5dde","isp":"\u7535\u4fe1"},{"ip":"180.123.94.155","port":4945,"expire_time":"2020-11-09 20:46:43","city":"\u5f90\u5dde","isp":"\u7535\u4fe1"},{"ip":"27.216.26.113","port":4945,"expire_time":"2020-11-09 20:48:46","city":"\u70df\u53f0","isp":"\u8054\u901a"},{"ip":"27.216.27.217","port":271,"expire_time":"2020-11-09 20:54:02","city":"\u70df\u53f0","isp":"\u8054\u901a"},{"ip":"218.91.1.49","port":14568,"expire_time":"2020-11-09 20:38:23","city":"\u626c\u5dde","isp":"\u7535\u4fe1"},{"ip":"223.215.5.79","port":14568,"expire_time":"2020-11-09 20:36:13","city":"\u9ec4\u5c71","isp":"\u7535\u4fe1"},{"ip":"116.138.243.178","port":940,"expire_time":"2020-11-09 20:43:17","city":"","isp":"\u8054\u901a"},{"ip":"36.6.134.34","port":7791,"expire_time":"2020-11-09 20:52:48","city":"\u6dee\u5317","isp":"\u7535\u4fe1"},{"ip":"117.87.209.107","port":62563,"expire_time":"2020-11-09 21:01:19","city":"\u5f90\u5dde","isp":"\u7535\u4fe1"},{"ip":"36.7.248.193","port":940,"expire_time":"2020-11-09 20:49:27","city":"\u5b89\u5e86","isp":"\u7535\u4fe1"},{"ip":"115.239.24.18","port":14568,"expire_time":"2020-11-09 20:52:40","city":"\u5609\u5174","isp":"\u7535\u4fe1"},{"ip":"124.113.243.241","port":14568,"expire_time":"2020-11-09 20:52:48","city":"\u5bbf\u5dde","isp":"\u7535\u4fe1"},{"ip":"114.97.92.28","port":14568,"expire_time":"2020-11-09 21:03:48","city":"\u868c\u57e0","isp":"\u7535\u4fe1"},{"ip":"49.68.185.83","port":271,"expire_time":"2020-11-09 20:36:10","city":"\u5f90\u5dde","isp":"\u7535\u4fe1"},{"ip":"223.215.174.38","port":14568,"expire_time":"2020-11-09 20:48:54","city":"\u6c60\u5dde","isp":"\u7535\u4fe1"},{"ip":"175.165.231.13","port":940,"expire_time":"2020-11-09 20:53:37","city":"\u76d8\u9526","isp":"\u8054\u901a"},{"ip":"122.188.244.192","port":271,"expire_time":"2020-11-09 20:37:18","city":"\u968f\u5dde","isp":"\u8054\u901a"},{"ip":"27.40.92.63","port":940,"expire_time":"2020-11-09 20:41:40","city":"\u60e0\u5dde","isp":"\u8054\u901a"},{"ip":"222.189.190.197","port":940,"expire_time":"2020-11-09 20:43:27","city":"\u626c\u5dde","isp":"\u7535\u4fe1"},{"ip":"221.231.56.120","port":7791,"expire_time":"2020-11-09 20:51:26","city":"\u76d0\u57ce","isp":"\u7535\u4fe1"},{"ip":"223.241.23.251","port":4945,"expire_time":"2020-11-09 20:47:28","city":"\u4eb3\u5dde","isp":"\u7535\u4fe1"},{"ip":"140.255.56.99","port":940,"expire_time":"2020-11-09 20:39:49","city":"\u6cf0\u5b89","isp":"\u7535\u4fe1"},{"ip":"125.123.141.105","port":4945,"expire_time":"2020-11-09 20:43:00","city":"\u5609\u5174","isp":"\u7535\u4fe1"},{"ip":"114.99.1.241","port":271,"expire_time":"2020-11-09 20:59:13","city":"\u94dc\u9675","isp":"\u7535\u4fe1"},{"ip":"183.165.29.35","port":7791,"expire_time":"2020-11-09 20:58:41","city":"\u5ba3\u57ce","isp":"\u7535\u4fe1"},{"ip":"114.227.11.93","port":4945,"expire_time":"2020-11-09 20:50:15","city":"\u5e38\u5dde","isp":"\u7535\u4fe1"},{"ip":"125.123.66.164","port":271,"expire_time":"2020-11-09 20:47:15","city":"\u5609\u5174","isp":"\u7535\u4fe1"},{"ip":"180.125.97.101","port":62563,"expire_time":"2020-11-09 20:44:43","city":"\u6dee\u5b89","isp":"\u7535\u4fe1"},{"ip":"117.60.239.6","port":4945,"expire_time":"2020-11-09 21:03:32","city":"\u6dee\u5b89","isp":"\u7535\u4fe1"},{"ip":"125.123.71.193","port":62563,"expire_time":"2020-11-09 20:53:18","city":"\u5609\u5174","isp":"\u7535\u4fe1"}]}


class ArticlespiderSpiderMiddleware(object):
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the spider middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_spider_input(response, spider):
        # Called for each response that goes through the spider
        # middleware and into the spider.

        # Should return None or raise an exception.
        return None

    def process_spider_output(response, result, spider):
        # Called with the results returned from the Spider, after
        # it has processed the response.

        # Must return an iterable of Request, dict or Item objects.
        for i in result:
            yield i

    def process_spider_exception(response, exception, spider):
        # Called when a spider or process_spider_input() method
        # (from other spider middleware) raises an exception.

        # Should return either None or an iterable of Response, dict
        # or Item objects.
        pass

    def process_start_requests(start_requests, spider):
        # Called with the start requests of the spider, and works
        # similarly to the process_spider_output() method, except
        # that it doesn’t have a response associated.

        # Must return only requests (not items).
        for r in start_requests:
            yield r

    def spider_opened(self, spider):
        spider.logger.info('Spider opened: %s' % spider.name)


class RandomUserAgentMiddlware(object):
    #随机更换user-agent
    def __init__(self, crawler):
        super(RandomUserAgentMiddlware, self).__init__()
        self.ua = UserAgent()
        self.ua_type = crawler.settings.get("RANDOM_UA_TYPE", "random")

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler)

    def process_request(self, request, spider):
        def get_ua():
            return getattr(self.ua, self.ua_type)

        request.headers.setdefault('User-Agent', get_ua())

class RandomProxyMiddleware(object):

    #动态设置ip代理
    def process_request(self, request, spider):
        # get_ip = GetIP()
        import random
        random_ip = random.choice(proxy_api['data'])['ip']
        print(random_ip)
        request.meta["proxy"] = random_ip


from selenium import webdriver
from scrapy.http import HtmlResponse
class JSPageMiddleware(object):

    #通过chrome请求动态网页
    def process_request(self, request, spider):
        if spider.name == "jobbole":
            # browser = webdriver.Chrome(executable_path="D:/Temp/chromedriver.exe")
            spider.browser.get(request.url)
            import time
            time.sleep(3)
            print ("访问:{0}".format(request.url))

            return HtmlResponse(url=spider.browser.current_url, body=spider.browser.page_source, encoding="utf-8", request=request)

# from pyvirtualdisplay import Display
# display = Display(visible=0, size=(800, 600))
# display.start()
#
# browser = webdriver.Chrome()
# browser.get()
