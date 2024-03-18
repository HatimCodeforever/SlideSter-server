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
from lida import Manager, TextGenerationConfig , llm
from io import BytesIO
import base64
from PIL import Image


OPENAI_API_KEY1 = os.getenv("OPENAI_API_KEY1")
OPENAI_API_KEY2 = os.getenv("OPENAI_API_KEY2")
TAVILY_API_KEY1 = os.getenv("TAVILY_API_KEY1")
TAVILY_API_KEY2 = os.getenv("TAVILY_API_KEY2")
HF_AUTH_TOKEN = os.getenv('HUGGINGFACE_API_KEY')
SDXL_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
LIDA = Manager(text_gen = llm("openai"), api_key= OPENAI_API_KEY2)
TEXTGEN_CONFIG_FOR_LIDA = TextGenerationConfig(n=1, temperature=0.5, model="gpt-3.5-turbo-1106", use_cache=True)
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
    IMAGE_GEN_MODEL = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        variant="fp16",
        torch_dtype=torch.float16,
        use_auth_token = HF_AUTH_TOKEN
    ).to("cuda")
    # SET SCHEDULER
    IMAGE_GEN_MODEL.scheduler = LCMScheduler.from_config(IMAGE_GEN_MODEL.scheduler.config)
    # LOAD LCM-LoRA
    IMAGE_GEN_MODEL.load_lora_weights("latent-consistency/lcm-lora-sdxl")

def generate_slide_titles(topic):
    client = OpenAI(api_key=OPENAI_API_KEY1)
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

def generate_point_info(topic, n_points, api_key_to_use):
    flag = 1 if api_key_to_use== 'first' else 2
    print(f'THREAD {flag} RUNNING...')
    openai_api_key = OPENAI_API_KEY1 if flag == 1 else OPENAI_API_KEY2
    client = OpenAI(api_key=openai_api_key)
    info_gen_prompt = """You will be given a list of topics and a corresponding list of number of points. Your task is to generate point-wise information on it for a powerpoint presentation. The points should be precise and plain sentences as that used in powerpoint presentations.

    Topics: {topics_list}
    Number of Points: {n_points_list}

    Generate information on these topics corresponding to the number of points in the list. Format the output as a JSON dictionary, where the keys are the topic name and the corresponding values are a list of points on that topic.

"""
    completion = client.chat.completions.create(
        model = 'gpt-3.5-turbo-1106',
        messages=[
            {
                'role':'user',
                'content': info_gen_prompt.format(topics_list=topic, n_points_list=n_points)
            }
        ],
        response_format = {'type':'json_object'},
        seed = 42,
    )

    output = ast.literal_eval(completion.choices[0].message.content)

    return output

def chat_generate_point_info(topic, n_points=5, api_key_to_use='first'):
    flag = 1 if api_key_to_use== 'first' else 2
    print(f'THREAD {flag} RUNNING...')
    openai_api_key = OPENAI_API_KEY1 if flag == 1 else OPENAI_API_KEY2
    client = OpenAI(api_key=openai_api_key)
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
    print("Images result", results['images_results'])
    image_results = results["images_results"]
    image_links = [i['original'] for i in image_results[:10]]
    return image_links


def generate_image(prompt):
    image_path = prompt + '.png'
    print('GENERATING IMAGE ON DEVICE TYPE:',DEVICE_TYPE)
    if DEVICE_TYPE == 'cuda':
        generator = torch.manual_seed(42)
        image = IMAGE_GEN_MODEL(
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
    client = OpenAI(api_key = OPENAI_API_KEY1)
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

def generate_point_info_from_document(topic, n_points, context, api_key_to_use):
    flag = 1 if api_key_to_use== 'first' else 2
    print(f'THREAD {flag} RUNNING...')
    openai_api_key = OPENAI_API_KEY1 if flag == 1 else OPENAI_API_KEY2
    client = OpenAI(api_key=openai_api_key)
    info_gen_prompt = """You will be given a list of topics and a corresponding list of number of points. You will also be provided with context from a document. Your task is to generate point-wise information on it for a powerpoint presentation using the provided context. The points should be precise and plain sentences as that used in powerpoint presentations. The number of points in each topic should be equal to the corresponding number of points in the list.

context : ```
{context}
```
Topics List : {topics_list}
number of points : {n_points}

Use the provided context to generate point-wise information. Format the output as a JSON dictionary, where the keys are the topic name and the corresponding values are a list of points on that topic.
"""
    completion = client.chat.completions.create(
        model = 'gpt-3.5-turbo-1106',
        messages=[
            {
                'role':'user',
                'content': info_gen_prompt.format(topics_list=topic, n_points=n_points, context= context)
            }
        ],
        response_format = {'type':'json_object'},
        seed = 42,
    )

    output = ast.literal_eval(completion.choices[0].message.content)

    return output


def generate_slide_titles_from_web(topic):
    client = OpenAI(api_key=OPENAI_API_KEY1)
    tavily_client = TavilyClient(api_key= TAVILY_API_KEY1)
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

def generate_point_info_from_web(topic, n_points, api_key_to_use):
    flag = 1 if api_key_to_use== 'first' else 2
    print(f'THREAD {flag} RUNNING...')
    openai_api_key = OPENAI_API_KEY1 if flag == 1 else OPENAI_API_KEY2
    tavily_api_key = TAVILY_API_KEY1 if flag == 1 else TAVILY_API_KEY2
    client = OpenAI(api_key = openai_api_key)
    tavily_client = TavilyClient(api_key = tavily_api_key)
    topic_str = ", ".join(topic)
    search_result = tavily_client.get_search_context(topic_str, search_depth="advanced", max_tokens=6000)
    
    info_gen_prompt = """You will be given a list of topics and a corresponding list of number of points. You will also be provided with context from internet. Your task is to generate point-wise information on it for a powerpoint presentation using the provided context from the internet. The points should be precise and plain sentences as that used in powerpoint presentations. The number of points in each topic should be equal to the corresponding number of points in the list.

Context from Internet: ```
{search_result}
```
Topics List : {topics_list}
number of points : {n_points}

Use the provided web context to generate point-wise information. Format the output as a JSON dictionary, where the keys are the topic name and the corresponding values are a list of points on that topic.
"""
    completion = client.chat.completions.create(
        model = 'gpt-3.5-turbo-1106',
        messages=[
            {
                'role':'user',
                'content': info_gen_prompt.format(topics_list=topic, n_points=n_points, search_result= search_result)
            }
        ],
        response_format = {'type':'json_object'},
        seed = 42,
    )

    output = ast.literal_eval(completion.choices[0].message.content)

    return output


def base64_to_image(base64_string):
    byte_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(byte_data))

def generate_summary(csv_file_path):
    summary = LIDA.summarize(csv_file_path, summary_method = 'default', textgen_config= TEXTGEN_CONFIG_FOR_LIDA)
    return summary

def generate_goals(summary, n_goals, persona):
    if persona:
        print('Generating goals with given persona...')
        goals = LIDA.goals(summary, n= n_goals, persona=persona, textgen_config= TEXTGEN_CONFIG_FOR_LIDA)
    else:
        print('Generating goals without persona...')
        goals = LIDA.goals(summary, n= n_goals, textgen_config= TEXTGEN_CONFIG_FOR_LIDA)
    return goals

def generate_visualizations(summary, goal, library='seaborn'):
    # libraries can be seaborn, matplotlib, ggplot, plotly, bokeh, altair
    charts = LIDA.visualize(summary=summary, goal= goal, textgen_config= TEXTGEN_CONFIG_FOR_LIDA, library=library)
    image_base64 = charts[0].raster
    img = base64_to_image(image_base64)
    return img, charts

def refine_visualizations(summary, code, instructions, library='seaborn'):
    edited_charts = LIDA.edit(code=code, summary=summary, instructions=instructions, library=library, textgen_config= TEXTGEN_CONFIG_FOR_LIDA)
    image_base64 = edited_charts[0].raster
    img = base64_to_image(image_base64)
    return img ,edited_charts

def generate_recommendations(code, summary, n_recc=1, library='seaborn'):
    recommended_charts =  LIDA.recommend(code=code, summary=summary, n=n_recc, library = library, textgen_config= TEXTGEN_CONFIG_FOR_LIDA)
    image_base64 = recommended_charts[0].raster
    img = base64_to_image(image_base64)
    return img, recommended_charts
