import json
import os
from tqdm import tqdm
from pytorch_transformers import BertTokenizer


def build_files(data_path, tokenized_data_path, num_pieces, full_tokenizer, min_length):
    with open(data_path, 'r', encoding='utf8') as f:
        print('reading lines')
        lines = json.load(f)
        lines = [line.replace('\n', ' [SEP] ') for line in lines]  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束
    all_len = len(lines)
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    for i in tqdm(range(num_pieces)):
        sublines = lines[all_len // num_pieces * i: all_len // num_pieces * (i + 1)]
        if i == num_pieces - 1:
            sublines.extend(lines[all_len // num_pieces * (i + 1):])  # 把尾部例子添加到最后一个piece
        sublines = [full_tokenizer.tokenize(line) for line in sublines if
                    len(line) > min_length]  # 只考虑长度超过min_length的句子
        sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]
        full_line = []
        for subline in sublines:
            full_line.append(full_tokenizer.convert_tokens_to_ids('[MASK]'))  # 文章开头添加MASK表示文章开始
            full_line.extend(subline)
            full_line.append(full_tokenizer.convert_tokens_to_ids('[CLS]'))  # 文章之间添加CLS表示文章结束
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'w') as f:
            for id in full_line:
                f.write(str(id) + ' ')
    print('finish')


def convert_raw_data_to_json(data_path, domain='Shuihu'):
    data_path = data_path + domain + '/'
    assert os.path.exists(data_path)
    datas = []
    files = os.listdir(data_path)
    for file in files:
        if domain + '.json' == file:
            exit('already converted!')
    for file in files:
        with open(data_path + file, 'r', encoding='utf8') as f:
            article = f.read()
            segments = article.split('\n\u3000\u3000')
            new_segments = [segment.replace('\n', '') for segment in segments]
            datas.append('\n'.join(new_segments))
    # print(datas)
    with open(data_path + domain + '.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(datas, ensure_ascii=False))


tokenizer = BertTokenizer.from_pretrained('D:/PythonProjects/语言模型/bert-base-chinese')
build_files(data_path='CrawlText/data/Shuihu/Shuihu.json',
            tokenized_data_path='CrawlText/data/Shuihu/Tokenized/',
            full_tokenizer=tokenizer,
            num_pieces=100,
            min_length=30)


