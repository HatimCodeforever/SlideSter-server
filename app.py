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
from utils import generate_slide_titles, generate_point_info, fetch_images_from_web

load_dotenv()


app = Flask(__name__)
passw = os.getenv("passw")
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
connection_string = f"mongodb+srv://hatim:{passw}@cluster0.f7or37n.mongodb.net/?retryWrites=true&w=majority"


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


@app.route("/generate-info")
def generate_info():
    print("Generating....")
    collection = MongoDB('ppt')
    doc = collection.find_one({'_id': ObjectId(session['info_id'])})
    topics = doc.get('titles')
    num_points = doc.get('points')
    # information = {
    # 'Introduction to Computer Vision': ['Computer vision is a field of study that focuses on enabling computers to see, recognize, and understand visual information.', 'It involves the use of various techniques such as image processing, pattern recognition, and machine learning algorithms.', 'Computer vision finds application in various domains including autonomous vehicles, robotics, healthcare, and surveillance systems.', 'Common tasks in computer vision include image classification, object detection, image segmentation, and image enhancement.', 'Python libraries like OpenCV and TensorFlow provide powerful tools and frameworks for implementing computer vision algorithms and applications.'],
    # 'The History of Computer Vision': ['The concept of computer vision dates back to the 1960s when researchers began exploring ways to enable computers to interpret visual information.', 'The development of computer vision was greatly influenced by advances in artificial intelligence and the availability of faster and more powerful hardware.', 'In the 1980s, computer vision techniques like edge detection and feature extraction gained popularity, leading to applications in fields like robotics and image recognition.', 'The 1990s saw significant progress in computer vision with the introduction of algorithms for object recognition, image segmentation, and motion detection.', 'In recent years, deep learning techniques, particularly convolutional neural networks(CNNs), have revolutionized computer vision by achieving state- of - the - art performance across a wide range of tasks.'],
    # 'Image Processing Techniques': ['Image processing techniques are used to enhance images and extract useful information from them.', 'Some common image processing techniques include image filtering, edge detection, image segmentation, and image morphology.', 'Image filtering can be used to remove noise from images or blur them.It involves applying a filter to each pixel in the image.', 'Edge detection techniques aim to find the boundaries of objects in an image by detecting abrupt changes in intensity.', 'Image segmentation techniques divide an image into regions or objects of interest based on their characteristics, such as color or texture.'],
    # 'Object Recognition and Classification': ['Object recognition and classification is the technology that allows computers to identify and categorize objects in images or videos.', 'It is an important component of many applications, such as self - driving cars, surveillance systems, and medical imaging.', 'Common approaches to object recognition and classification include deep learning - based methods, such as convolutional neural networks(CNNs).', 'These methods often require large annotated datasets for training, where objects in the images / videos are manually labeled with their corresponding classes.', 'Object recognition and classification can be challenging due to variations in lighting, viewpoint, occlusion, and object appearance.'],
    # 'Object Tracking in Computer Vision': ['Object tracking is the process of locating and following a specific object in a video or sequence of images.', 'It is a challenging task in computer vision due to factors such as viewpoint changes, occlusions, and appearance variations.', 'Object tracking algorithms can be categorized into two main types: online and offline.', 'Online tracking algorithms operate in real - time and track objects in a frame - by - frame manner.', 'Offline tracking algorithms require access to the entire video sequence and can perform more complex tracking tasks.'],
    # 'Segmentation and Image Analysis': ['Segmentation is the process of dividing an image into multiple segments to simplify the image and make it easier to analyze.', 'Image analysis refers to the process of extracting meaningful information from an image, such as object recognition or measuring certain properties of the image.', 'Segmentation techniques can be based on various criteria, such as color, texture, or shape.', 'Image analysis can be used in various fields, such as medical imaging, surveillance, and computer vision.', 'Both segmentation and image analysis are important techniques in computer vision for understanding and extracting information from images.'], 
    # 'Applications of Computer Vision in Robotics': ['Computer Vision is used in robotics for object detection and recognition, allowing robots to identify and interact with objects in their environment.', 'Computer Vision can be used to track and follow objects or people, enabling robots to perform tasks such as surveillance or guiding people in public spaces.', 'Computer Vision is used in navigation systems for robots, allowing them to perceive and interpret their surroundings to plan and execute their movements.', 'Computer Vision is used for robot localization and mapping, allowing robots to build a representation of their environment and determine their position within it.', 'Computer Vision is used in robot - assisted surgeries, providing surgeons with enhanced visual information and real - time feedback during procedures.'],
    # 'Deep Learning in Computer Vision': ['Deep learning is a subfield of machine learning that focuses on training artificial neural networks with multiple layers to solve complex problems in computer vision.', 'Convolutional Neural Networks(CNNs) are commonly used in deep learning for computer vision tasks, as they are designed to effectively process and extract features from visual data.', 'Transfer learning is widely used in deep learning for computer vision, where pre - trained models on large datasets are used as a starting point for training new models on smaller, specific datasets.', 'Deep learning models for computer vision have achieved remarkable results in image classification, object detection, image segmentation, and image generation tasks.', 'Popular deep learning frameworks like TensorFlow, PyTorch, and Keras have extensive support for computer vision tasks and provide pre - built architectures and tools to facilitate deep learning in this domain.'],
    # 'Challenges in Computer Vision': ['Limited labeled data for training models.', 'Difficulty in handling variations in lighting and perspective.', 'Complexity in identifying and recognizing objects in cluttered scenes.', 'Challenges in accurately segmenting and extracting object boundaries.', 'Handling occlusions and partial visibility of objects.'],
    # 'Future Trends in Computer Vision': ['The use of computer vision in various industries such as healthcare, retail, and automotive is expected to continue growing rapidly.', 'Advances in deep learning algorithms have significantly improved the accuracy and efficiency of computer vision systems.', 'Real-time object detection and tracking will become more common, allowing for a wide range of applications such as autonomous vehicles and surveillance systems.', 'Computer vision technology will increasingly be integrated with other emerging technologies such as augmented reality and virtual reality.', 'The use of computer vision for facial recognition and emotion detection is likely to become more prevalent, raising privacy and ethical concerns.']
    # }
    information = {}
    for topic in topics:
      output = generate_point_info(topic=topic, n_points=num_points)
      information[topic] = list(output.values())[0]

    print(information)
    keys = list(information.keys())
    return jsonify({"keys": keys, "information": information})

@app.route('/fetch-images', methods=['POST'])
def fetch_images():
    data = request.get_json()
    topics = data.get('topic')
    all_images = {}
    for topic in topics:
        images = fetch_images_from_web(topic)
        all_images[topic] = images

    return jsonify({"images": all_images})

if __name__ == "__main__":
  app.run(debug=True)
