import openai
from openai import OpenAI
from tavily import TavilyClient
import os
import ast
import torch
import time

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_slide_titles(topic):
    client = OpenAI()
    title_suggestion_prompt = """Generate 10 compelling slide titles for a PowerPoint Presentation on the given topic. Format the output in JSON, with each key representing the slide number and its corresponding value being the slide title. Be creative and ensure that the titles cover key aspects of the topic, providing a comprehensive overview.

Topic = {topic}
"""
    completion = client.chat.completions.create(
        model = 'gpt-3.5-turbo-1106',
        messages=[
            {
                'role':'user',
                'content': title_suggestion_prompt.format(topic=topic)
            }
        ]
    )

    output = ast.literal_eval(completion.choices[0].message.content)
    return output

def generate_point_info(topic, n_points=5):
    client = OpenAI()
    info_gen_prompt = """You will be given a topic and your task is to generate {n_points} points of information on it. The points should be precise and plain sentences. Format the output as a JSON dictionary, where the key is the topic name and the value is a list of points.

Topic : {topic}
"""
    completion = client.chat.completions.create(
        model = 'gpt-3.5-turbo-1106',
        messages=[
            {
                'role':'user',
                'content': info_gen_prompt.format(topic=topic, n_points=n_points)
            }
        ]
    )

    output = ast.literal_eval(completion.choices[0].message.content)

    return output

def fetch_images_from_web(topic):
    tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
    search_results = tavily_client.search(topic, search_depth="advanced",include_images=True)
    images = search_results['images']
    return images
