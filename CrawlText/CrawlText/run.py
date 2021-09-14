from scrapy import cmdline
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--crawler', type=str, default='CCTV_News')

args = parser.parse_args()

print(cmdline.execute('scrapy crawl {}'.format(args.crawler).split(' ')))
