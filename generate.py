import argparse
from flask import Flask, request
from utils import fast_sample_sequence
import transformers
import pytorch_transformers
import json

app = Flask(__name__)


def generate(input_context):
    input_ids = tokenizer.encode(input_context)
    out = fast_sample_sequence(model, input_ids, args.generate_length)
    generate_text = tokenizer.convert_ids_to_tokens(out)
    text = []
    for token in generate_text:
        if token != '[UNK]':
            text += [token]
        elif token == '[SEP]':
            text += ['\n']

    return ''.join(text)


@app.route('/generate', methods=['POST', 'GET'])
def index():
    question = str(request.args.get("question"))
    response = generate(question)
    return json.dumps({'status': '200 OK', 'response': response}, ensure_ascii=False)


def run_cmd():
    print('欢迎使用狗屁不通生成器！')
    while True:
        print('=' * 80)
        input_context = input('请输入一个故事开头，或输入exit退出:')
        if input_context == 'exit':
            break
        response = generate(input_context)
        print('狗屁不通生成器生成结果：')
        print('=' * 80)
        print(response)


parser = argparse.ArgumentParser()
parser.add_argument('--gpt_pretrained_path', default='model/final_model/', type=str)
parser.add_argument('--bert_pretrained_path', default='D:/PythonProjects/语言模型/bert-base-chinese', type=str)
parser.add_argument('--generate_length', default=150, type=int)
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--cmd', action='store_true', default=False,
                    help='是否使用命令行模式, 否则启动Flask服务')

args = parser.parse_args()

model = transformers.GPT2LMHeadModel.from_pretrained(args.gpt_pretrained_path).to(args.device)
tokenizer = pytorch_transformers.BertTokenizer.from_pretrained(args.bert_pretrained_path)
if args.cmd:
    run_cmd()
else:
    app.run(debug=True, threaded=True, host='0.0.0.0', port=2333)


