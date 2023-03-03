from flask import Flask, render_template, request, url_for, redirect,session,jsonify
import pandas as pd
from py.rekomendation import RekomendasiBuku
import pickle
import json

app = Flask(__name__)
app = Flask(__name__,template_folder='temp')

re = pickle.load(open("model.sav", 'rb'))

@app.route('/')
def index():
  return render_template("index.html")

@app.route('/rekomendasi',methods=['POST'])
def rekomendasi():
  data = request.form.to_dict(flat=False)

  title = re.user_read(data['Form_Search'][0])
  user = re.result(sorted(list(title['Title'])))
  print(title)
  print(user)

  buku = user.drop(columns=['categories'])
  buku = buku.fillna("")
  buku['category'] = buku['category'].apply(lambda x:", ".join(x))

  buku['category'] = buku['category'].apply(lambda x:x.replace("'", " " ))
  buku['Title'] = buku['Title'].apply(lambda x:x.replace("'", " " ))

  databuku = {}
  no = 0

  for d in range(len(buku)):
    u = buku.iloc[[d]]
    databuku[no] = {
      'judul':u.Title.values[0],
      'kat':u.category.values[0],
      'gambar':u.image.values[0]
    }
    no+=1
  
  print(databuku)

  data = {
    'prediksi':databuku,
    'evaluasi':re.intra_list_similarity(user.index,user)
  }

  return str(json.dumps(data))

if __name__=='__main__':
  app.run()