from scrapy import cmdline

print(cmdline.execute('scrapy crawl CCTV_News'.split(' ')))
