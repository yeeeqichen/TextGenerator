import scrapy
from scrapy import Request
import re
from ..items import ShuihuItem
import os


class ShuihuSpider(scrapy.Spider):
    name = 'Shuihu'
    allowed_domains = ['purepen.com']
    start_urls = ['http://purepen.com/shz/']

    def start_requests(self):
        for url in self.start_urls:
            yield Request(url=url,
                          callback=self.parse_catalogue)

    def parse_catalogue(self, response):
        # print(response.text)
        urls = response.xpath('//a/@href').extract()
        for url in urls[1:]:
            yield Request(url=self.start_urls[0] + url,
                          callback=self.parse)

    def parse(self, response):
        item = ShuihuItem()
        title = response.xpath('//b/text()').extract_first()
        content = response.xpath('//font/text()').extract_first()
        title = title.replace('《水浒传》 ', '').replace(' ', '')
        section, name = re.match('(第.{1,4}回)(.*)', title).groups()
        item['title'] = section + '_' + name
        item['content'] = content
        yield item
