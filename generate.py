import argparse
from flask import Flask, request
from utils import fast_sample_sequence
import transformers
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

    return ''.join(text).replace('Ġ', ' ')


@app.route('/generate', methods=['POST', 'GET'])
def index():
    question = str(request.args.get("question"))
    response = generate(question)
    return json.dumps({'status': '200 OK', 'response': response}, ensure_ascii=False)


def run_cmd():
    print('Welcome to TextGenerator！')
    while True:
        print('=' * 80)
        input_context = input('please enter a prefix text to start generating; or enter "exit" to stop:')
        if input_context == 'exit':
            break
        response = generate(input_context)
        print('generating result：')
        print('=' * 80)
        print(response)


parser = argparse.ArgumentParser()
parser.add_argument('--gpt_pretrained_path', default='model/final_model', type=str)
parser.add_argument('--tokenizer_path', default='distilgpt2', type=str)
parser.add_argument('--generate_length', default=150, type=int)
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--cmd', action='store_true', default=False,
                    help='是否使用命令行模式, 否则启动Flask服务')

args = parser.parse_args()

model = transformers.GPT2LMHeadModel.from_pretrained(args.gpt_pretrained_path).to(args.device)
tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.tokenizer_path)
if args.cmd:
    run_cmd()
else:
    app.run(debug=True, threaded=True, host='0.0.0.0', port=2333)


