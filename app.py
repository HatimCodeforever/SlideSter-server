from flask import Flask, request, jsonify, session, json
from pymongo import MongoClient
import bcrypt
import jwt
from datetime import datetime, timedelta
import os
from bson import ObjectId
from dotenv import load_dotenv
from flask import request, jsonify
import openai
from openai import OpenAI
import re
import ast
from utils import generate_slide_titles, generate_point_info, fetch_images_from_web, chat_generate_point_info
import torch
import time

load_dotenv()


app = Flask(__name__)
passw = os.getenv("passw")
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
connection_string = f"mongodb+srv://hatim:{passw}@cluster0.f7or37n.mongodb.net/?retryWrites=true&w=majority"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
openai.api_key = os.getenv('OPENAI_API_KEY')
    

def MongoDB(collection_name):
    client = MongoClient(connection_string)
    db = client.get_database("SlideSter")
    records = db.get_collection(collection_name)
    return records


def generate_token(user_id):
    payload = {"user_id": user_id, "exp": datetime.utcnow() + timedelta(hours=1)}
    token = jwt.encode(payload, app.config["SECRET_KEY"], algorithm="HS256")
    return token


def create_session(user_email):
    session["user_email"] = user_email


# records = MongoDB('register')


@app.route("/adduser", methods=["POST"])
def adduser():
    new_record = request.json
    email = new_record["email"]
    existing_user = MongoDB('register').find_one({"email": email})
    if existing_user:
        response = {"message": "exists"}
        return jsonify(response)

    salt = bcrypt.gensalt()
    new_record["password"] = bcrypt.hashpw(new_record["password"].encode("utf-8"), salt)
    result = MongoDB('register').insert_one(new_record)

    if result.inserted_id:
        token = generate_token(str(result.inserted_id))
        response = {"message": "success", "token": token}
        return jsonify(response)
    else:
        response = {"message": "failed"}
        return jsonify(response)


@app.route("/home")
def home():
    return "hello"


@app.route("/profile", methods=["GET"])
def profile():
    user_email = session.get("user_email")
    response2 = MongoDB('register').find_one({"email": user_email})
    del response2["_id"]
    del response2["password"]
    return jsonify(response2)


@app.route("/login", methods=["POST"])
def login():
    new_record = request.json
    user = MongoDB('register').find_one({"email": new_record["email"]})
    if user:
        if bcrypt.checkpw(new_record["password"].encode("utf-8"), user["password"]):
            token = generate_token(str(user["_id"]))
            response = {"message": "success", "token": token}
            create_session(str(user["email"]))
            return jsonify(response)
        else:
            response = {"message": "password"}
            return jsonify(response)
    else:
        response = {"message": "username"}
        return jsonify(response)


@app.route("/model1", methods=["POST"])
def model1():
    data = request.json
    titles = data.get("titles")
    points = data.get("points")
    print(titles)
    print(points)
    ppt_data = {
      "titles": titles,
      "points": points
    }
    collection = MongoDB('ppt')
    result=collection.insert_one(ppt_data)
    session['info_id'] = str(result.inserted_id)
    response = {"message": True}
    return jsonify(response)


@app.route("/logout", methods=["GET"])
def logout():
    session.clear()
    response = {"message": "success"}
    return jsonify(response)


@app.route("/suggest-titles", methods=["POST"])
def suggest_titles():
    data = request.get_json()
    domain = data.get("domain")
    topic = data.get("topic")
    output = generate_slide_titles(topic)
    response_list = list(output.values())
    print(response_list)
    # final_suggestion_list = [
    # 'Introduction', 'Applications', 'Types of Machine Learning',
    # 'Supervised Learning', 'Unsupervised Learning', 'Reinforcement Learning',
    # 'Data Preprocessing', 'Model Evaluation', 'Challenges and Limitations',
    # 'Future Trends'
    # ]
    response = {"message": response_list}
    return jsonify(response)

@app.route('/generate-new-info', methods=['POST'])
def generate_new_info():
    data = request.get_json()
    topic = data.get('topic')
    information = generate_point_info(topic=topic)
    print(information)
    keys = list(information.keys())
    return jsonify({"key": keys, "information": information})

slide_number = 3
tools = [
    {
        'type': 'function',
        'function':{
            'name': 'generate_information',
            'description': 'generates and adds information when given a topic and a slide number, ask the user for all the specified arguments.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'topic': {
                        'type': 'string',
                        'description': 'The topic on which the information is to be generated. For Example: Introduction to Machine Learning'
                    },
                    'slide_number' :{
                        'type': 'integer',
                        'description': 'The number of slide at which the information is to be added.'
                    },
                    'n_points' :{
                        'type': 'integer',
                        'description': 'The number of points of information to be generated, default is 5.'
                    }
                },
                'required': ['topic', 'slide_number', 'n_points']
            }
        }
    },
]

available_tools = {
    'generate_information': chat_generate_point_info
}

@app.route("/generate-info")
def generate_info():
    print("Generating....")
    collection = MongoDB('ppt')
    doc = collection.find_one({'_id': ObjectId(session['info_id'])})
    topics = doc.get('titles')
    num_points = doc.get('points')
    information = {
    'Introduction to Computer Vision': ['Computer vision is a field of study that focuses on enabling computers to see, recognize, and understand visual information.', 'It involves the use of various techniques such as image processing, pattern recognition, and machine learning algorithms.', 'Computer vision finds application in various domains including autonomous vehicles, robotics, healthcare, and surveillance systems.', 'Common tasks in computer vision include image classification, object detection, image segmentation, and image enhancement.', 'Python libraries like OpenCV and TensorFlow provide powerful tools and frameworks for implementing computer vision algorithms and applications.'],
    'The History of Computer Vision': ['The concept of computer vision dates back to the 1960s when researchers began exploring ways to enable computers to interpret visual information.', 'The development of computer vision was greatly influenced by advances in artificial intelligence and the availability of faster and more powerful hardware.', 'In the 1980s, computer vision techniques like edge detection and feature extraction gained popularity, leading to applications in fields like robotics and image recognition.', 'The 1990s saw significant progress in computer vision with the introduction of algorithms for object recognition, image segmentation, and motion detection.', 'In recent years, deep learning techniques, particularly convolutional neural networks(CNNs), have revolutionized computer vision by achieving state- of - the - art performance across a wide range of tasks.'],
    }
    # information = {}
    # for topic in topics:
    #     output = generate_point_info(topic=topic, n_points=num_points)
    #     information[topic] = list(output.values())[0]
    # all_images = {}
    all_images = {'Introduction to Machine Learning': ['https://onpassive.com/blog/wp-content/uploads/2020/12/AI-01-12-2020-860X860-Kumar.jpg', 'https://www.flexsin.com/blog/wp-content/uploads/2019/05/1600_900_machine_learning.jpg', 'https://www.globaltechcouncil.org/wp-content/uploads/2021/06/Machine-Learning-Trends-That-Will-Transform-The-World-in-2021-1.jpg', 'http://csr.briskstar.com/Content/Blogs/ML Blog.jpg', 'https://s3.amazonaws.com/media.the-next-tech.com/wp-content/uploads/2021/01/19132558/Top-6-Machine-Learning-Trends-you-should-watch-in-2021.jpg'], 'Future Trends in Machine Learning': ['https://onpassive.com/blog/wp-content/uploads/2020/12/AI-01-12-2020-860X860-Kumar.jpg', 'https://tenoblog.com/wp-content/uploads/2019/03/Machine-Learning-Technologies.jpg', 'https://www.flexsin.com/blog/wp-content/uploads/2019/05/1600_900_machine_learning.jpg', 'https://tai-software.com/wp-content/uploads/2020/01/machine-learning.jpg', 'https://www.techolac.com/wp-content/uploads/2021/07/robot-1536x1024.jpg']}
    # for topic in topics:
    #     images = fetch_images_from_web(topic)
    #     all_images[topic] = images
    # print(information)
    keys = list(information.keys())
    # print(all_images)
    client = OpenAI()
    assistant = client.beta.assistants.create(
        name="SLIDESTER",
        instructions="You are a helpful assistant. Use the tools provided to you to help the user.",
        model="gpt-3.5-turbo-0613",
        tools =  tools
    )
    session['assistant_id'] = assistant.id
    print('ASSITANT:',assistant)
    return jsonify({"keys": keys, "information": information, "images": all_images})

def wait_on_run(run_id, thread_id):
    client = OpenAI()
    while True:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id,
        )
        print('RUN STATUS', run.status)
        time.sleep(0.5)
        if run.status in ['failed', 'completed', 'requires_action']:
            return run
        
def get_tool_result(thread_id, run_id, tools_to_call):
    tools_outputs = []
    for tool in tools_to_call:
        output = None
        tool_call_id = tool.id
        tool_name = tool.function.name
        tool_args = tool.function.arguments
        print('TOOL CALLED:',tool_name)
        print('ARGUMENTS:', tool_args)

        if tool_name == 'generate_information':
            tool_to_call = available_tools.get(tool_name)
            topic = json.loads(tool_args)['topic']
            n_points = json.loads(tool_args)['n_points']
            output = tool_to_call(topic= topic, n_points= n_points)

            print('OUTPUT:',output)

        if output:
            tools_outputs.append({'tool_call_id': tool_call_id, 'output': output})
    return tools_outputs
        
@app.route('/chatbot-route', methods=['POST'])
def chatbot_route():
    data = request.get_json()
    print(data)
    query = data.get('userdata', '')
    if query:         
        client = OpenAI()
        assistant_id = session['assistant_id']
        print('ASSISTANT ID',assistant_id)
        thread = client.beta.threads.create()
        print('THREAD ID', thread.id)
        
        message = client.beta.threads.messages.create(
            thread_id= thread.id,
            role="user",
            content= query,
        )
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=session['assistant_id'],
        )
        

        run = wait_on_run(run.id, thread.id)

        if run.status == 'failed':
            print(run.error)
        elif run.status == 'requires_action':
            run = get_tool_result(thread.id, run.id, run.required_action.submit_tool_outputs.tool_calls)
            # run = wait_for_run_completion(thread.id, run.id)
            print('HELLO', run[0]['output'])
        
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        print(messages.data[0].content[0])
        chatbot_reply = messages.data[0].content[0].text.value
        output = run[0]
        keys = list(output['output'].keys())
        # Return the chatbot response
        return jsonify({'chatbotResponse': chatbot_reply,'function': ['generate_info'],'key': keys, 'information': output['output']})
    else:
        return jsonify({'error': 'User message not provided'}), 400
    
    
    

if __name__ == "__main__":
  app.run(debug=True)
