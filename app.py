from flask import Flask, render_template, request, url_for, redirect,session,jsonify
import pandas as pd

app = Flask(__name__)
app = Flask(__name__,template_folder='temp')

@app.route('/')
def index():
  return render_template("index.html")

if __name__=='__main__':
  app.run()