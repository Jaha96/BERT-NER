from flask import Flask,request,jsonify
from flask_cors import CORS

from bert import Ner

import MeCab
import subprocess
import textwrap

cmd = 'echo `mecab-config --dicdir`"/mecab-ipadic-neologd"'
path = (subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]).decode('utf-8')
m = MeCab.Tagger("-d {0}".format(path))
def mecab_normalize(str_text):
  n = m.parseToNode(str_text)
  sentence = ""
  while n:
    # print(n.surface, "\t", n.feature)
    sentence += n.surface + " "
    n = n.next
  sentence = ' '.join(sentence.split())
  return sentence


app = Flask(__name__)
CORS(app)

model = Ner("./out_base/")

@app.route("/predict",methods=['POST'])
def predict():
    text = request.json["text"]
    result = []
    try:
        str_txts = textwrap.wrap(text, 500)
        for t in str_txts:
            r = model.predict(t)
            result += r
        # out = model.predict(text)
        return jsonify({"result":result})
    except Exception as e:
        print(e)
        return jsonify({"result":"Model Failed"})

if __name__ == "__main__":
    app.run('0.0.0.0',port=8000)