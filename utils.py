import openai
from openai import OpenAI
from tavily import TavilyClient
import os
import ast
import torch
from diffusers import DiffusionPipeline, LCMScheduler
import requests
import io
from PIL import Image
import torch
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader, PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from serpapi import GoogleSearch


openai.api_key = os.getenv("OPENAI_API_KEY")
# os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
HF_AUTH_TOKEN = os.getenv('HUGGINGFACE_API_KEY')
SDXL_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
GOOGLE_SERP_API_KEY = os.getenv('GOOGLE_SERP_API_KEY')
VECTORDB_FILE_PATH = 'faiss_index'


DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".py": TextLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".pdf": PyPDFLoader
}

DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
ENCODE_KWARGS = {'normalize_embeddings': True} # set True to compute cosine similarity
EMBEDDINGS = HuggingFaceBgeEmbeddings(
    model_name= EMBEDDING_MODEL_NAME,
    model_kwargs={'device': DEVICE_TYPE },
    encode_kwargs=ENCODE_KWARGS
)

if DEVICE_TYPE=='cuda':
    image_gen_model = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        variant="fp16",
        torch_dtype=torch.float16,
        use_auth_token = HF_AUTH_TOKEN
    ).to("cuda")
    # SET SCHEDULER
    image_gen_model.scheduler = LCMScheduler.from_config(image_gen_model.scheduler.config)
    # LOAD LCM-LoRA
    image_gen_model.load_lora_weights("latent-consistency/lcm-lora-sdxl")

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
        ],
        response_format = {'type':'json_object'},
        seed = 42,
    )

    output = ast.literal_eval(completion.choices[0].message.content)
    return output

def generate_point_info(topic, n_points=5):
    client = OpenAI()
    info_gen_prompt = """You will be given a list of topics and a corresponding list of number of points. Your task is to generate point-wise information on it lfor a powerpoint presentation. The points should be precise and plain sentences as that used in powerpoint presentations. Format the output as a JSON dictionary, where the keys are the topic name and the corresponding values are a list of points on that topic.

    Topics: {topics_list}
    Number of Points: {n_points_list}
"""
    completion = client.chat.completions.create(
        model = 'gpt-3.5-turbo-1106',
        messages=[
            {
                'role':'user',
                'content': info_gen_prompt.format(topic=topic, n_points=n_points)
            }
        ],
        response_format = {'type':'json_object'},
        seed = 42,
    )

    output = ast.literal_eval(completion.choices[0].message.content)

    return output

def chat_generate_point_info(topic, n_points=5):
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
        ],
        response_format = {'type':'json_object'},
        seed = 42,
    )

    output = ast.literal_eval(completion.choices[0].message.content)

    return output

def fetch_images_from_web(topic):
    params = {
    "q": topic,
    "engine": "google_images",
    "ijn": "0",
    "api_key":  GOOGLE_SERP_API_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    image_results = results["images_results"]
    image_links = [i['original'] for i in image_results[:10]]
    return image_links


def generate_image(prompt):
    image_path = prompt + '.png'
    print('GENERATING IMAGE ON DEVICE TYPE:',DEVICE_TYPE)
    if DEVICE_TYPE == 'cuda':
        generator = torch.manual_seed(42)
        image = image_gen_model(
            prompt=prompt, num_inference_steps=4, generator=generator, guidance_scale=1.0
        ).images[0]

        image.save(image_path)
    
    else:
        headers = {"Authorization": "Bearer "+ HF_AUTH_TOKEN}
        payload = {'inputs': prompt}
        response = requests.post(SDXL_API_URL, headers=headers, json = payload)
        print(response)
        image_bytes = response.content
        print(image_bytes[:100])
        image = Image.open(io.BytesIO(image_bytes)) 
        image.save(image_path)

    return image_path


def ingest(file_path):
    file_extension = os.path.splitext(file_path)[1]
    loader_class = DOCUMENT_MAP.get(file_extension)
    if loader_class:
        loader = loader_class(file_path)
    else:
        raise ValueError("Document type is not supported")
    loader = loader_class(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)

    print('CONVERTING TEXTS TO EMBEDDINGS...')
    vector_db = FAISS.from_documents(texts, EMBEDDINGS)
    print('VECTOR DATABASE CREATED')
    vector_db.save_local(VECTORDB_FILE_PATH)

    return VECTORDB_FILE_PATH

def generate_slide_titles_from_document(topic, context):
    client = OpenAI()
    info_gen_prompt = """Generate 5 most relvant and compelling slide titles for a PowerPoint Presentation on the given topic and based on the given context. \
    It should cover the major aspects of the context \
    Format the output in JSON, with each key representing the slide number and its corresponding value being the slide title. \
    Be creative and ensure that the titles cover key aspects of the topic, providing a comprehensive overview.

    Topic = {topic}

    Context = {context}
    """
    completion = client.chat.completions.create(
        model = 'gpt-3.5-turbo-1106',
        messages=[
            {
                'role':'user',
                'content': info_gen_prompt.format(topic= topic, context = context)
            }
        ],
        response_format = {'type':'json_object'},
        seed = 42,

    )

    output = ast.literal_eval(completion.choices[0].message.content)

    return output

def generate_point_info_from_document(topic, n_points, context):
    client = OpenAI()
    info_gen_prompt = """You will be given a topic and some context. Your task is to generate {n_points} points of information using the context. The points should be precise and plain sentences. Format the output as a JSON dictionary, where the key is the topic name and the value is a list of points.

Topic : {topic}

context : {context}
"""
    completion = client.chat.completions.create(
        model = 'gpt-3.5-turbo-1106',
        messages=[
            {
                'role':'user',
                'content': info_gen_prompt.format(topic=topic, n_points=n_points, context= context)
            }
        ],
        response_format = {'type':'json_object'},
        seed = 42,
    )

    output = ast.literal_eval(completion.choices[0].message.content)

    return output


def generate_slide_titles_from_web(topic):
    client = OpenAI()
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    search_result = tavily_client.get_search_context(topic, search_depth="advanced", max_tokens=4000)
    
    info_gen_prompt = """Generate 10 most relevant and compelling slide titles for a PowerPoint Presentation on the given topic, \ 
    based on the information provided from the web. \
    It should cover the major aspects of the search results. \
    Format the output in JSON, with each key representing the slide number and its corresponding value being the slide title. \
    Be creative and ensure that the titles cover key aspects of the topic, providing a comprehensive overview.

    Topic = {topic}

    Search Result = {search_result}
    """
    completion = client.chat.completions.create(
        model = 'gpt-3.5-turbo-1106',
        messages=[
            {
                'role':'user',
                'content': info_gen_prompt.format(topic= topic, search_result = search_result)
            }
        ],
        response_format = {'type':'json_object'},
        seed = 42,

    )

    output = ast.literal_eval(completion.choices[0].message.content)

    return output

def generate_point_info_from_web(topic, n_points):
    client = OpenAI()
    tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
    search_result = tavily_client.get_search_context(topic, search_depth="advanced", max_tokens=4000)
    
    info_gen_prompt = """You will be given a topic and search results from the internet. Your task is to generate {n_points} points of information using the search results. The points should be precise and plain sentences. Format the output as a JSON dictionary, where the key is the topic name and the value is a list of points.

Topic : {topic}

Search Results : {search_result}
"""
    completion = client.chat.completions.create(
        model = 'gpt-3.5-turbo-1106',
        messages=[
            {
                'role':'user',
                'content': info_gen_prompt.format(topic=topic, n_points=n_points, search_result= search_result)
            }
        ],
        response_format = {'type':'json_object'},
        seed = 42,
    )

    output = ast.literal_eval(completion.choices[0].message.content)

    return output





