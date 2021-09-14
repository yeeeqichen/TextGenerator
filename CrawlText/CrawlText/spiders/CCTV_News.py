import scrapy
from scrapy_splash import SplashRequest
from scrapy import Request
from ..items import CCTVNewsItem


class CctvNewsSpider(scrapy.Spider):
    name = 'CCTV_News'
    allowed_domains = ['english.cctv.com']
    start_urls = ['http://english.cctv.com/']

    def start_requests(self):
        scripts = """
                    function main(splash, args)
                        splash:go(args.url)
                        splash:wait(0.3)
                        local cur_height = splash:evaljs("document.body.scrollTop")
                        local scrollHeight = splash:evaljs("document.body.scrollHeight")
                        local prev_height = 0
                        local torrent = 10
                        local lag_cnt = 0
                        while(cur_height < scrollHeight)
                        do
                            splash:evaljs("window.scrollTo(0, document.body.scrollHeight)")
                            splash:wait(0.3)
                            prev_height = cur_height
                            cur_height = splash:evaljs("document.body.scrollTop")
                            scrollHeight = splash:evaljs("document.body.scrollHeight")
                            print(cur_height, scrollHeight)
                            if prev_height == cur_height then
                                lag_cnt = lag_cnt + 1
                                if lag_cnt == torrent then
                                    break
                                end
                            end
                        end
                        return {
                            html = splash:html()
                        }
                    end
                """
        for url in self.start_urls:
            yield SplashRequest(url=url,
                                callback=self.parse_first,
                                endpoint='execute',
                                args={
                                    'lua_source': scripts,
                                    'timeout': 90
                                })

    def parse_first(self, response):
        xpath_str = '//body//div[@id="page_body"]//div[@class="English19035_ind04"]//li//h3[@class="title"]/a/@href'
        urls = response.xpath(xpath_str).extract()
        print(len(urls))
        for url in urls:
            yield SplashRequest(url, callback=self.parse_second)

    def parse_second(self, response):
        item = CCTVNewsItem()
        xpath_str_title = '//body//div[@id="page_body"]//div[@class="title_area"]//h1/text()'
        title = response.xpath(xpath_str_title).extract_first()
        if title is None:
            return
        # print(title)
        xpath_str_text = '//body//div[@id="page_body"]//div[@class="content_area"]//p/text()'
        text = response.xpath(xpath_str_text).extract()
        article = '\n'.join(text)
        item['title'] = title
        item['content'] = article
        yield item
        # print(text)
        # print(response.text)
