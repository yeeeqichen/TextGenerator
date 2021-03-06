# TextGenerator
![](https://img.shields.io/badge/scrapy-%20%3D%3D%202.5.0-informational)   ![](https://img.shields.io/badge/Python-%20%3E%3D%20%203.7.0-success)  

### Description
An easy-to-use framework of TextGenerator, supports different domains, different languages text generation!

If you already have some corpus for training, just follow the Quick Start to train your own TextGenerator in 3 steps!

Or we provide you a crawler to crawl your own data from internet
### Features

- A crawler built with scrapy, you can easily get your train corpus based on your domain
- A model train script for you to train your own TextGenerator
- Provide many ways of generation, including: cmd-line mode and web-server mode

### Dependencies

- scrapy == 2.5.0
- scrapy_splash == 0.7.2
- transformers == 4.10.0
- torch == 1.8.1

### Quick Start

#### Step 1
put your domain-specific raw data into data/domain-name/raw, then run the data preprocess script in data/:
```shell
python3 convert_rawdata.py \
  --tokenizer_path <the file or url to initialize the GPT2Tokenizer> \
  --domain_name <the domain-name of your own corpus, defalut set to 'domain-name'>
```
Tips: 
- an example file of raw data is put in data/domain-name/raw
- it will create a new directory 'data/tokenized/' , which contains the tokenized data for further training.
- this step (as well as step 3) needs to initialize the GPT2Tokenizer, with hugging-face url, i.e 'gpt2distil', for more information, please refer to [hugging face](https://huggingface.co/)

#### Step 2
then run the train script to train your own TextGenerator model:
```shell
python3 main.py \
  --tokenized_data_path <the generated data path in previous step, i.e data/domain-name/tokenized> \
  --pretrained_model <the GPT pretrained model path, you could specify a directory in your pc or a url provided by hugging-face, i.e 'uer/gpt2-chinese-cluecorpussmall'> \
  --device <the gpu device you want to use, support multi-gpus, i.e 0,1,2> \
  --output_dir <the directory to store the model after training, default set to model/>
```
Tips: for more detailed parameter setting, please refer to main.py


#### Step 3
The final step, run the generating script to start your own TextGenerator, have fun!
```shell
python3 generate.py \
  --gpt_pretrained_path <the directory you store your TextGenerator path> \
  --tokenzier_path <the directory or url for a pretrained tokenizer> \
  --device <which device should the model run on>
```
Tips: to use command-line mode, add '--cmd' to the command above


### Crawl your own corpus
In this repo, we provide several crawlers for you to crawl training corpus from Internet 

you could crawl your own corpus by customize the spider setting

To enable the crawler to work normally, you need to install following dependencies:

- scrapy
- scrapy_splash

also, a splash service provided by docker is also needed, run the following command:
```shell
docker pull scrapinghub/splash
docker run -p 8050:8050 scrapinghub/splash
```
Tips: to learn more about docker, you could refer to [here](http://get.daocloud.io/)

Finally, run the python script to start your crawler
```shell
python3 CrawlText/CrawlText/run.py \
  --crawler <specify a crawler , default set to 'CCTV_News', for chinese corpus, please set to 'Shuihu'>
```



### Generate Result

#### Chinese
???????????????
![](https://github.com/yeeeqichen/Pictures/blob/master/shz_generate.png?raw=true)
????????????????????????
![](https://github.com/yeeeqichen/Pictures/blob/master/shz_daiyu.png?raw=true)
#### English
CCTV_News
![](https://github.com/yeeeqichen/Pictures/blob/master/CCTV_result.png?raw=true)






