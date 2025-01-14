import os
import subprocess
import textwrap
import uuid
import copy

import MeCab
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from bert import Ner
from image_processing import start_processing

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')

# Make directory if "uploads" folder not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

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

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = Ner("./out_base/")

@app.route("/")                   # at the end point /
def hello():                      # call method hello
    return "Hello World!" 


def filter_result(model_results):
  new_results = []
  for result in model_results:
    if "tag" in result and ("PER" in result["tag"]):
      new_results.append(result)
  return new_results

#Upload
@app.route('/api/upload',methods=['GET','POST'])
def uploadFile():
    if request.method == 'POST':
      response = []
      if 'imagefile' not in request.files:
          return jsonify({"result":"No files uploaded"})
          # if file and allowed_file(file.filename):

      files = request.files.getlist('imagefile')
      for file in files:
          print(file)
          filename = secure_filename(file.filename)

          # Gen GUUID File Name
          fileExt = filename.split('.')[1]
          autoGenFileName = uuid.uuid4()

          newFileName = str(autoGenFileName) + '.' + fileExt
          file_path = os.path.join(app.config['UPLOAD_FOLDER'], newFileName)
          file.save(file_path)

          text = start_processing(file_path)
          return_text = copy.deepcopy(text)
          str_text = mecab_normalize(text)
          model_result = []
          try:
            str_txts = textwrap.wrap(str_text, 500)
            for t in str_txts:
                r = model.predict(t)
                model_result += r
            model_result = filter_result(model_result)
            response.append({"text": return_text, "ner": model_result})
          except Exception as e:
            print(e)
            return jsonify({"result":"Model Failed"})

          
      return jsonify({"result":response})
    else:
        return jsonify({"result":"Please post"})

@app.route("/predict",methods=['POST'])
def predict():
    text = request.json["text"]
    str_text = mecab_normalize(text)
    result = []
    try:
        str_txts = textwrap.wrap(str_text, 500)
        for t in str_txts:
            r = model.predict(t)
            result += r
        # out = model.predict(text)
        model_result = filter_result(result)
        return jsonify({"result":model_result})
    except Exception as e:
        print(e)
        return jsonify({"result":"Model Failed"})

if __name__ == "__main__":
    app.run('0.0.0.0',port=8000)
