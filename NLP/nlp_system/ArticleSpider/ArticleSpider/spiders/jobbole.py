# -*- coding: utf-8 -*-
import re
import scrapy
from scrapy.http import Request
from urllib import parse
import json

from ArticleSpider.items import JobBoleArticleItem, ArticleItemLoader

from ArticleSpider.utils.common import get_md5
from selenium import webdriver
# from scrapy.xlib.pydispatch import dispatcher
from scrapy import signals

class JobboleSpider(scrapy.Spider):
    name = "jobbole"
    allowed_domains = ['news.cnblogs.com']
    start_urls = ['http://news.cnblogs.com/']


    # def __init__(self):
    #     self.browser = webdriver.Chrome(executable_path="D:/Temp/chromedriver.exe")
    #     super(JobboleSpider, self).__init__()
    #     dispatcher.connect(self.spider_closed, signals.spider_closed)
    #
    # def spider_closed(self, spider):
    #     #当爬虫退出的时候关闭chrome
    #     print ("spider closed")
    #     self.browser.quit()

    #收集伯乐在线所有404的url以及404页面数
    handle_httpstatus_list = [404]

    def __init__(self, **kwargs):
        self.fail_urls = []
        # dispatcher.connect(self.handle_spider_closed, signals.spider_closed)

    def handle_spider_closed(self, spider, reason):
        self.crawler.stats.set_value("failed_urls", ",".join(self.fail_urls))

    def parse(self, response):
        """
        1. 获取文章列表页中的文章url并交给scrapy下载后并进行解析
        2. 获取下一页的url并交给scrapy进行下载， 下载完成后交给parse
        """
        #解析列表页中的所有文章url并交给scrapy下载后并进行解析
        if response.status == 404:
            self.fail_urls.append(response.url)
            self.crawler.stats.inc_value("failed_url")

        post_nodes = response.css('#news_list .news_block')
        for post_node in post_nodes:
            image_url = post_node.css('.entry_summary a img::attr(src)').extract_first("")
            if image_url.startswith("//"):
                image_url = "https:" + image_url
            post_url = post_node.css('h2 a::attr(href)').extract_first("")
            yield Request(url=parse.urljoin(response.url, post_url), meta={"front_image_url": image_url},
                          callback=self.parse_detail)
            break

        #提取下一页并交给scrapy进行下载
        # next_url = response.xpath("//a[contains(text(), 'Next >')]/@href").extract_first("")
        # if next_url:
        #     yield Request(url=parse.urljoin(response.url, next_url), callback=self.parse)

    def parse_detail(self, response):
        match_re = re.match(".*?(\d+)", response.url)
        if match_re:
            post_id = match_re.group(1)

            """
            article_item = JobboleArticleItem()
            title = response.css("#news_title a::text").extract_first("")
            # title = response.xpath('//*[@id="news_title"]//a/text()')
            create_data = response.css("#news_info .time::text").extract_first("")
            match_re = re.match(".*?(\d+.*)", create_data)
            if match_re:
                create_date = match_re.group(1)
                # create_date = response.xpath('//*[@id="news_info"]//*[@class="time"]/text()')
            content = response.css("#news_content").extract()[0]
            # content = response.xpath('//*[@id="news_content"]').extract()[0]
            tag_list = response.css(".news_tags a::text").extract()
            # tag_list = response.xpath('//*[@class="news_tags"]//a/text()').extract()
            tags = ",".join(tag_list)
            """

            '''
            同步请求代码，在并发要求不是很高时可以采用
            post_id = match_re.group(1)
            html = requests.get(parse.urljoin(response.url, "/NewsAjax/GetAjaxNewsInfo?contentId={}".format(post_id)))
            j_data = json.loads(html.text)
            '''

            """
            article_item["title"] = title
            article_item["create_date"] = create_date
            article_item["content"] = content
            article_item["tags"] = tags
            article_item["url"] = response.url
            # 报错：ValueError:Missing scheme in request url:h
            # 上述报错原因：对于图片下载的字段一定要使用list类型，故[response.meta.get("front_image_url", "")]
            if response.meta.get("front_image_url", ""):
                article_item["front_image_url"] = [response.meta.get("front_image_url", "")]
            else:
                article_item["front_image_url"] = []
            """

            item_loader = ArticleItemLoader(item=JobBoleArticleItem(), response=response)
            item_loader.add_css("title", "#news_title a::text")
            item_loader.add_css("create_date", "#news_info .time::text")
            item_loader.add_css("content", "#news_content")
            item_loader.add_css("tags", ".news_tags a::text")
            item_loader.add_value("url", response.url)
            if response.meta.get("front_image_url", []):
                item_loader.add_value("front_image_url", response.meta.get("front_image_url", []))

            # article_item = item_loader.load_item()
            print(parse.urljoin(response.url, "/NewsAjax/GetAjaxNewsInfo?contentId={}".format(post_id)))
            yield Request(url=parse.urljoin(response.url, "/NewsAjax/GetAjaxNewsInfo?contentId={}".format(post_id)),
                          meta={"article_item": item_loader, "url": response.url}, callback=self.parse_nums)

    def parse_nums(self, response):
        j_data = json.loads(response.text)
        item_loader = response.meta.get("article_item", "")

        praise_nums = j_data["DiggCount"]
        fav_nums = j_data["TotalView"]
        comment_nums = j_data["CommentCount"]

        item_loader.add_value("praise_nums", j_data["DiggCount"])
        item_loader.add_value("fav_nums", j_data["TotalView"])
        item_loader.add_value("comment_nums", j_data["CommentCount"])
        item_loader.add_value("url_object_id", get_md5(response.meta.get("url", "")))
        '''
        article_item["praise_nums"] = praise_nums
        article_item["fav_nums"] = fav_nums
        article_item["comment_nums"] = comment_nums
        article_item["url_object_id"] = common.get_md5(article_item["url"])
        '''

        article_item = item_loader.load_item()

        yield article_item