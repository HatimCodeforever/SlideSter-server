from flask import Flask, request, jsonify, session, json,send_file,make_response
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
from utils import generate_slide_titles, generate_point_info, fetch_images_from_web, chat_generate_point_info, generate_image, ingest, generate_slide_titles_from_document, generate_point_info_from_document, EMBEDDINGS
import torch
import time
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from werkzeug.utils import secure_filename

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
    doc = data.get("doc")
    # print(titles)
    # print(points)
    print("doc status",doc)
    ppt_data = {
      "titles": titles,
      "points": points,
      "doc" : doc
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
    # final_suggestion_list = [
    #     'Introduction', 'Applications', 'Types of Machine Learning',
    #     'Supervised Learning', 'Unsupervised Learning', 'Reinforcement Learning',
    #     'Data Preprocessing', 'Model Evaluation', 'Challenges and Limitations',
    #     'Future Trends'
    #     ]
    
    domain = request.form.get('domain')
    topic = request.form.get('topic')

    if 'file' not in request.files:
       output = generate_slide_titles(topic)
       response_list = list(output.values())
       print(response_list)
       response = {"message": response_list,"doc":False}
       return jsonify(response)
    else:
        file = request.files['file']
        print("print file ",file)
        local_path = 'pdf-file'
        file.save(os.path.join(local_path, secure_filename(file.filename)))
        file_path = 'pdf-file/'+ secure_filename(file.filename)
        vectordb_file_path = ingest(file_path)
        vector_db= FAISS.load_local(vectordb_file_path, EMBEDDINGS)
        query1 = topic
        query2 = "Technology or architecture"
        session["vectordb_file_path"]=vectordb_file_path
        docs1 = vector_db.similarity_search(query1)
        docs2 = vector_db.similarity_search(query2)

        all_docs = docs1 + docs2

        context = [doc.page_content for doc in all_docs]
        output = generate_slide_titles_from_document(topic, context)
        response_list = list(output.values())

        response = {"message": response_list,"doc":True}
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
            'description': 'Generates information when given a topic and a slide number',
            'parameters': {
                'type': 'object',
                'properties': {
                    'topic': {
                        'type': 'string',
                        'description': 'The topic on which the information is to be generated. For Example: Introduction to Machine Learning'
                    },
                    'slide_number' :{
                        'type': 'string',
                        'description': 'The number of the slide at which the information is to be added.'
                    },
                    'n_points' :{
                        'type': 'string',
                        'description': 'The number of points of information to be generated, default is 5.'
                    }
                },
                'required': ['topic', 'slide_number', 'n_points']
            }
        }
    },
    {
        'type': 'function',
        'function':{
            'name': 'generate_image',
            'description': 'Generates images when given an image generation prompt',
            'parameters': {
                'type': 'object',
                'properties': {
                    'prompt': {
                        'type': 'string',
                        'description': 'An appropriate prompt for the image generation model following a specific format for example, Astronaut in a jungle, cold color palette, muted colors, detailed, 8k'
                    },
                    'slide_number' :{
                        'type': 'string',
                        'description': 'The number of the slide at which the generated image is to be added.'
                    },
                },
                'required': ['prompt', 'slide_number']
            }
        }
    },
    {
        'type': 'function',
        'function':{
            'name': 'change_style',
            'description': 'Change the style (color or font-size) of the text when given a color and font size',
            'parameters': {
                'type': 'object',
                'properties': {
                    'text_color': {
                        'type': 'string',
                        'description': 'The color of transform the text into. Example red, green, etc.'
                    },
                    'font_size': {
                        'type': 'string',
                        'description': 'The size of the text.'
                    }
                },
                'required': ['text_color', 'font_size']
            }
        }
    },
]

available_tools = {
    'generate_information': chat_generate_point_info,
    'generate_image': generate_image
}

@app.route("/generate-info")
def generate_info():
    print("Generating....")
    collection = MongoDB('ppt')
    doc = collection.find_one({'_id': ObjectId(session['info_id'])})
    topics = doc.get('titles')
    num_points = doc.get('points')
    doc = doc.get('doc')
    print('doc',doc)
    if not doc:
        # information = {
        # 'Introduction to Computer Vision': ['Computer vision is a field of study that focuses on enabling computers to see, recognize, and understand visual information.', 'It involves the use of various techniques such as image processing, pattern recognition, and machine learning algorithms.', 'Computer vision finds application in various domains including autonomous vehicles, robotics, healthcare, and surveillance systems.', 'Common tasks in computer vision include image classification, object detection, image segmentation, and image enhancement.', 'Python libraries like OpenCV and TensorFlow provide powerful tools and frameworks for implementing computer vision algorithms and applications.'],
        # 'The History of Computer Vision': ['The concept of computer vision dates back to the 1960s when researchers began exploring ways to enable computers to interpret visual information.', 'The development of computer vision was greatly influenced by advances in artificial intelligence and the availability of faster and more powerful hardware.', 'In the 1980s, computer vision techniques like edge detection and feature extraction gained popularity, leading to applications in fields like robotics and image recognition.', 'The 1990s saw significant progress in computer vision with the introduction of algorithms for object recognition, image segmentation, and motion detection.', 'In recent years, deep learning techniques, particularly convolutional neural networks(CNNs), have revolutionized computer vision by achieving state- of - the - art performance across a wide range of tasks.'],
        # }
        information = {}
        for topic in topics:
            output = generate_point_info(topic=topic, n_points=num_points)
            information[topic] = list(output.values())[0]
        print(information)

        all_images = {'Introduction to Machine Learning': ['https://onpassive.com/blog/wp-content/uploads/2020/12/AI-01-12-2020-860X860-Kumar.jpg', 'https://www.flexsin.com/blog/wp-content/uploads/2019/05/1600_900_machine_learning.jpg', 'https://www.globaltechcouncil.org/wp-content/uploads/2021/06/Machine-Learning-Trends-That-Will-Transform-The-World-in-2021-1.jpg', 'http://csr.briskstar.com/Content/Blogs/ML Blog.jpg', 'https://s3.amazonaws.com/media.the-next-tech.com/wp-content/uploads/2021/01/19132558/Top-6-Machine-Learning-Trends-you-should-watch-in-2021.jpg'], 'Future Trends in Machine Learning': ['https://onpassive.com/blog/wp-content/uploads/2020/12/AI-01-12-2020-860X860-Kumar.jpg', 'https://tenoblog.com/wp-content/uploads/2019/03/Machine-Learning-Technologies.jpg', 'https://www.flexsin.com/blog/wp-content/uploads/2019/05/1600_900_machine_learning.jpg', 'https://tai-software.com/wp-content/uploads/2020/01/machine-learning.jpg', 'https://www.techolac.com/wp-content/uploads/2021/07/robot-1536x1024.jpg']}
        # all_images = {}
        for topic in topics:
            images = fetch_images_from_web(topic)
            all_images[topic] = images
        keys = list(information.keys())
        client = OpenAI()
        assistant = client.beta.assistants.create(
            name="SLIDESTER",
            instructions="You are a helpful assistant. Please use the functions provided to you appropriately to help the user.",
            model="gpt-3.5-turbo-0613",
            tools =  tools
        )
        thread = client.beta.threads.create()
        session['assistant_id'] = assistant.id
        session['thread_id'] = thread.id
        
        print('ASSITANT INITIALISED: ',assistant)
        return jsonify({"keys": keys, "information": information, "images": all_images})
    else:
        information = {}
        vectordb_file_path = session["vectordb_file_path"]
        vector_db = FAISS.load_local(vectordb_file_path, EMBEDDINGS)
        for topic in topics:
            rel_docs = vector_db.similarity_search(topic)
            time.sleep(25)
            context = [doc.page_content for doc in rel_docs]
            output = generate_point_info_from_document(topic=topic, n_points=num_points, context=context)
            information[topic] = list(output.values())[0]
        all_images = {}
        for topic in topics:
            images = fetch_images_from_web(topic)
            all_images[topic] = images
        keys = list(information.keys())
        client = OpenAI()
        assistant = client.beta.assistants.create(
            name="SLIDESTER",
            instructions="You are a helpful assistant. Please use the functions provided to you appropriately to help the user.",
            model="gpt-3.5-turbo-0613",
            tools =  tools
        )
        thread = client.beta.threads.create()
        session['assistant_id'] = assistant.id
        session['thread_id'] = thread.id

        print('ASSITANT INITIALISED: ',assistant)

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
    all_tool_name = []
    for tool in tools_to_call:
        output = None
        tool_call_id = tool.id
        tool_name = tool.function.name
        tool_args = tool.function.arguments
        tool_to_call = available_tools.get(tool_name)
        print('TOOL CALLED:',tool_name)
        print('ARGUMENTS:', tool_args)
        all_tool_name.append(tool_name)
        if tool_name == 'generate_information':
            topic = json.loads(tool_args)['topic']
            n_points = json.loads(tool_args)['n_points']
            output = tool_to_call(topic= topic, n_points= n_points)
            print('OUTPUT:',output)
            if output:
                tools_outputs.append({'generate_info_output': output })
        elif tool_name == 'generate_image':
            prompt = json.loads(tool_args)['prompt']
            print('Generating image...')
            image_path = generate_image(prompt)
            print('Image generated and saved at path:',image_path)
            tools_outputs.append({'generate_image_output': image_path })
        
    return tools_outputs,all_tool_name
        
@app.route('/chatbot-route', methods=['POST'])
def chatbot_route():
    data = request.get_json()
    print(data)
    query = data.get('userdata', '')
    if query:         
        client = OpenAI()
        assistant_id = session['assistant_id']
        print('ASSISTANT ID',assistant_id)
        thread_id = session['thread_id']
        print('THREAD ID', thread_id)
        
        message = client.beta.threads.messages.create(
            thread_id= thread_id,
            role="user",
            content= query,
        )
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=session['assistant_id'],
        )
        run = wait_on_run(run.id, thread_id)

        if run.status == 'failed':
            print(run.error)
        elif run.status == 'requires_action':
            all_output,tool = get_tool_result(thread_id, run.id, run.required_action.submit_tool_outputs.tool_calls)
            # run = wait_for_run_completion(thread.id, run.id)
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        
        if "generate_information" in tool:
            print('Generating information')
            print(all_output[0]['generate_info_output'])
            chatbot_reply = "Yes sure! Your information has been added on your current Slide!"
            keys = list(all_output[0]['generate_info_output'])
            all_images= {}
            images = fetch_images_from_web(keys[0])
            all_images[keys[0]] = images
            response = {'chatbotResponse': chatbot_reply, "images": all_images,'function_name': 'generate_information','key': keys, 'information': all_output[0]['generate_info_output']} 
            
        elif "generate_image" in tool:
            print('generate_image')
            image_path = all_output[0]['generate_image_output']
            chatbot_reply = "Yes sure! Your image has been added on your current Slide!"
            image_url = f"/send_image/{image_path}"
            # Create a response object to include both image and JSON data
            response = {'chatbotResponse': chatbot_reply,'function_name': 'generate_image','image_url': image_url}
        return jsonify(response)
    else:
        return jsonify({'error': 'User message not provided'}), 400
    

@app.route('/send_image/<image_path>', methods=['GET'])
def send_image(image_path):
    return send_file(image_path, mimetype='image/jpeg')

if __name__ == "__main__":
  app.run(debug=True)