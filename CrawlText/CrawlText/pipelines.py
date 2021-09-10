# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import os


class CrawltextPipeline:
    def process_item(self, item, spider):
        target_dir = os.path.abspath('../') + '/data/' + spider.name + '/'
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        with open(target_dir + item['title'] + '.txt', 'w', encoding='utf-8') as f:
            f.write(item['content'])
        return item



