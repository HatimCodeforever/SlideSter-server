from flask import Flask,request,jsonify,session,json
from pymongo import MongoClient
import bcrypt 
import jwt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from flask import request, jsonify

import openai
import re
import ast

load_dotenv()



app = Flask(__name__)
passw = os.getenv("passw")
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")
connection_string = f"mongodb+srv://hatim:{passw}@cluster0.f7or37n.mongodb.net/?retryWrites=true&w=majority"
def MongoDB():
  client = MongoClient(connection_string)
  db = client.get_database('SlideSter')
  records = db.register
  return records

def generate_token(user_id):
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=1)
    }
    token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')
    return token

def create_session(user_email):
    session['user_email'] = user_email

records = MongoDB()
@app.route("/adduser",methods=['POST'])
def adduser():
  new_record = request.json
  email = new_record['email']
  existing_user = MongoDB().find_one({'email': email})
  if existing_user:
      response = {'message': 'exists'}
      return jsonify(response)

  salt = bcrypt.gensalt()
  new_record['password'] = bcrypt.hashpw(new_record['password'].encode('utf-8'), salt)
  result = MongoDB().insert_one(new_record)
    
  if result.inserted_id:
    token = generate_token(str(result.inserted_id))
    response = {'message': 'success', 'token': token}
    return jsonify(response)
  else:
    response = {'message': 'failed'}
    return jsonify(response)


@app.route("/home")
def home():
  return 'hello' 


@app.route("/profile", methods=['GET'])
def profile():
  user_email = session.get('user_email')
  response2 = MongoDB().find_one({'email': user_email})   
  del response2['_id']
  del response2['password']
  return jsonify(response2)

@app.route("/login", methods=['POST'])
def login():
    new_record = request.json
    user = MongoDB().find_one({'email': new_record['email']})
    if user:
        if bcrypt.checkpw(new_record['password'].encode('utf-8'), user['password']):
            token = generate_token(str(user['_id']))
            response = {'message': 'success', 'token': token}
            create_session(str(user['email']))
            return jsonify(response)
        else:
            response = {'message': 'password'}
            return jsonify(response)
    else:
        response = {'message': 'username'}
        return jsonify(response)


@app.route("/model1", methods=['POST'])
def model1():
    data = request.json
    titles = data.get('titles')
    points = data.get('points')
    print(titles)
    print(points)
    response = {'message': True}
    return jsonify(response)

@app.route("/logout", methods=['GET'])
def logout():
  session.clear()
  response = {'message': 'success'}
  return jsonify(response)

@app.route("/suggest-titles", methods=['POST'])
def suggest_titles():
  data = request.get_json()
  domain = data.get('domain')
  topic = data.get('topic')
  openai.api_key = os.getenv('OPENAI_API_KEY')
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages= [
      {
          "role": "system",
          "content": '''Create a list of 10 slide titles for a PowerPoint presentation. You will be given a topic, and your task is to suggest slide titles that could be included in the presentation. For instance, you might suggest titles like 'Introduction' or 'Advantages.' Your goal is to return a list of slide topics that should be relevant and informative for the given presentation topic. Refrain from adding any other irrelevant information or text besides the list in the response.
          Template:
          ```suggested_titles = [{suggested titles}]``` Please follow this template.
          '''
      },
      {
          "role": "user",
          "content": f'''{topic}'''
      }
    ],
    max_tokens=450,
    frequency_penalty=0,
    presence_penalty=0
    )
  
  rep=response.choices[0].message.content
  re_list = re.sub('suggested_titles\s=\s',"", rep)
  final_suggestion_list = ast.literal_eval(re_list)
  print(final_suggestion_list)
  response = {"message": final_suggestion_list}
  return jsonify(response)

if  __name__=="__main__":
    app.run(debug=True)