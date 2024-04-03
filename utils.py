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
from reportlab.lib.pagesizes import letter
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import mm


OPENAI_API_KEY1 = os.getenv("OPENAI_API_KEY1")
OPENAI_API_KEY2 = os.getenv("OPENAI_API_KEY2")
TAVILY_API_KEY1 = os.getenv("TAVILY_API_KEY1")
TAVILY_API_KEY2 = os.getenv("TAVILY_API_KEY2")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY2

HF_AUTH_TOKEN = os.getenv('HUGGINGFACE_API_KEY')
SDXL_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
LIDA = Manager(text_gen = llm("openai"))
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

def generate_point_info(main_topic, topic, n_points, api_key_to_use):
    flag = 1 if api_key_to_use== 'first' else 2
    print(f'THREAD {flag} RUNNING...')
    openai_api_key = OPENAI_API_KEY1 if flag == 1 else OPENAI_API_KEY2
    client = OpenAI(api_key=openai_api_key)
    info_gen_prompt = """You will be given a main topic. You will also be provided with a list of sub-topics and a corresponding list of number of points. Your task is to generate point-wise information on it for a powerpoint presentation. The points should be precise and plain sentences as that used in powerpoint presentations.

    Main Topic: {main_topic}
    Sub-Topics: {topics_list}
    Number of Points: {n_points_list}

    Generate information on these topics corresponding to the number of points in the list. Format the output as a JSON dictionary, where the keys are the topic name and the corresponding values are a list of points on that topic.

"""
    completion = client.chat.completions.create(
        model = 'gpt-3.5-turbo-1106',
        messages=[
            {
                'role':'user',
                'content': info_gen_prompt.format(main_topic= main_topic, topics_list=topic, n_points_list=n_points)
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
    image_results = results['images_results']
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

def generate_point_info_from_document(main_topic, topic, n_points, context, api_key_to_use):
    flag = 1 if api_key_to_use== 'first' else 2
    print(f'THREAD {flag} RUNNING...')
    openai_api_key = OPENAI_API_KEY1 if flag == 1 else OPENAI_API_KEY2
    client = OpenAI(api_key=openai_api_key)
    info_gen_prompt = """You will be given a main topic. Additionally, youw will be given a list of sub-topics and a corresponding list of number of points. You will also be provided with context from a document. Your task is to generate point-wise information on it for a powerpoint presentation using the provided context. The points should be precise and plain sentences as that used in powerpoint presentations. The number of points in each topic should be equal to the corresponding number of points in the list.

context : ```
{context}
```
Main topic : {main_topic}
Topics List : {topics_list}
number of points : {n_points}

Use the provided context to generate point-wise information. Format the output as a JSON dictionary, where the keys are the topic name and the corresponding values are a list of points on that topic.
"""
    completion = client.chat.completions.create(
        model = 'gpt-3.5-turbo-1106',
        messages=[
            {
                'role':'user',
                'content': info_gen_prompt.format(main_topic= main_topic, topics_list=topic, n_points=n_points, context= context)
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

def generate_point_info_from_web(main_topic, topic, n_points, api_key_to_use):
    flag = 1 if api_key_to_use== 'first' else 2
    print(f'THREAD {flag} RUNNING...')
    openai_api_key = OPENAI_API_KEY1 if flag == 1 else OPENAI_API_KEY2
    tavily_api_key = TAVILY_API_KEY1 if flag == 1 else TAVILY_API_KEY2
    client = OpenAI(api_key = openai_api_key)
    tavily_client = TavilyClient(api_key = tavily_api_key)
    topic_str = ", ".join(topic)
    search_result = tavily_client.get_search_context(main_topic+':'+topic_str, search_depth="advanced", max_tokens=6000)
    
    info_gen_prompt = """You will be given a main topic as well as a list of topics and a corresponding list of number of points. You will also be provided with context from internet. Your task is to generate point-wise information on it for a powerpoint presentation using the provided context from the internet. The points should be precise and plain sentences as that used in powerpoint presentations. The number of points in each topic should be equal to the corresponding number of points in the list.

Context from Internet: ```
{search_result}
```
Main Topic: {main_topic}
Topics List : {topics_list}
number of points : {n_points}

Use the provided web context to generate point-wise information. Format the output as a JSON dictionary, where the keys are the topic name and the corresponding values are a list of points on that topic.
"""
    completion = client.chat.completions.create(
        model = 'gpt-3.5-turbo-1106',
        messages=[
            {
                'role':'user',
                'content': info_gen_prompt.format(main_topic=main_topic, topics_list=topic, n_points=n_points, search_result= search_result)
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
    if not persona==None:
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

def edit_visualizations(summary, code, instructions, library='seaborn'):
    edited_charts = LIDA.edit(code=code, summary=summary, instructions=instructions, library=library, textgen_config= TEXTGEN_CONFIG_FOR_LIDA)
    image_base64 = edited_charts[0].raster
    img = base64_to_image(image_base64)
    return img ,edited_charts

def recommend_visualizations(summary, code, n_recc=1, library='seaborn'):
    recommended_charts =  LIDA.recommend(code=code, summary=summary, n=n_recc, library = library, textgen_config= TEXTGEN_CONFIG_FOR_LIDA)
    all_images= []
    for chart in recommended_charts:
        print("single chart:-------------------",chart)
        image_base64 = chart.raster
        image = base64_to_image(image_base64)
        all_images.append(image)
    return all_images, recommended_charts

def generate_question_bank(context, n_questions=10):
    client = OpenAI(api_key=OPENAI_API_KEY1)
    info_gen_prompt = """
    Your task is to create a question bank based on the context provided from a PowerPoint presentation. Generate a total of {n_questions} questions using only the provided context. Only use the provided context to generate the questions. Each question should have its answer directly present in the context itself.

    CONTEXT:
    ```{context}```

    Ensure that the output is in valid JSON format, with keys representing the question numbers and values containing the questions generated by you.
    """
    completion = client.chat.completions.create(
        model = 'gpt-3.5-turbo-1106',
        messages=[
            {
                'role':'user',
                'content': info_gen_prompt.format(n_questions= n_questions, context=context)
            }
        ],
        response_format = {'type':'json_object'},
        seed = 42,
    )

    output = ast.literal_eval(completion.choices[0].message.content)
    
    return output

def generate_notes(context):
    client = OpenAI(api_key=OPENAI_API_KEY1)
    info_gen_prompt = """
    You have been provided with content extracted from a PowerPoint presentation. Your task is to generate comprehensive notes for the faculty member who will be delivering the presentation. Ensure that the notes are well-structured and easily understandable for the faculty's reference during and after the presentation.

    Your notes should include:

    1. Brief Explanations: Provide concise explanations for all important topics discussed in the slides.
    2. Jargon Definitions: Define any specialized terminology or jargon used in the presentation. Organize these definitions as a dictionary, where the jargon terms are the keys and its meaning are the values.
    3. Core Concept Explanations: Explain difficult or core concepts in a manner that is accessible to the faculty member. Clearly identify each section or topic covered in the presentation and provide in depth explanation of each. Organize these as a dictionary, where the keys are the sections and the values are the in-depth explanation of that section. The explanation should be detailed and more than 100 words for all the concepts.
    4. Examples: Include relevant examples to illustrate key points and aid in understanding.

    CONTEXT:
    ```{context}```

    Ensure that the output is in valid JSON format, with keys corresponding to "brief_explanation" (a dictionary of explanations),"jargons_meaning" (a dictionary of jargons), "concepts_explanation" (a dictionary of explanations), and "examples" (a dictionary of examples).
    """
    completion = client.chat.completions.create(
        model = 'gpt-3.5-turbo-1106',
        messages=[
            {
                'role':'user',
                'content': info_gen_prompt.format(context=context)
            }
        ],
        response_format = {'type':'json_object'},
        seed = 42,
    )

    output = ast.literal_eval(completion.choices[0].message.content)
    return output

def generate_question_bank_pdf(pdf_file_path, main_topic , question_bank):
    # Register Unicode fonts
    pdfmetrics.registerFont(TTFont('DejaVuSansCondensed', 'Fonts/DejaVuSansCondensed.ttf'))

    # Create a PDF document
    pdf = SimpleDocTemplate(pdf_file_path, pagesize=letter)

    # Define styles for different headings and content
    styles = {
        'Heading1': ParagraphStyle(name='Heading1', fontName='DejaVuSansCondensed', fontSize=16, spaceAfter=16, spaceBefore=16, bold=True),
        'Heading2': ParagraphStyle(name='Heading2', fontName='DejaVuSansCondensed', fontSize=14, spaceAfter=14, spaceBefore=14),
        'Heading3': ParagraphStyle(name='Heading3', fontName='DejaVuSansCondensed', fontSize=12, spaceAfter=12, spaceBefore=12),
        'Normal': ParagraphStyle(name='Normal', fontName='DejaVuSansCondensed', fontSize=10, spaceAfter=8, spaceBefore=8),
        'URL': ParagraphStyle(name='URL', textColor=colors.blue, underline=True, spaceAfter=8),
    }

    # Build the PDF document
    content = [
        Paragraph("SLIDESTER", styles['Heading1']),
        # Image('logo/apple-icon.png', width=140, height=237),
        Spacer(1, 0.01*inch),
        Paragraph("Disclaimer: This content is generated by AI.", styles['Heading3']),
        Spacer(1, 0.5*inch),
        Paragraph("Topic: "+main_topic, styles['Heading1']),
        Spacer(1, 0.05*inch),
        Paragraph("Question Bank", styles['Heading3']),
        Spacer(1, 0.02*inch),
    ]

    for key,value in question_bank.items():
        content.append(Paragraph(key +" "+ value, styles['Normal']))

    pdf.build(content, onFirstPage=add_page_number, onLaterPages=add_page_number)

def generate_notes_pdf(pdf_file_path, main_topic , notes):
    # Register Unicode fonts
    pdfmetrics.registerFont(TTFont('DejaVuSansCondensed', 'Fonts/DejaVuSansCondensed.ttf'))

    # Create a PDF document
    pdf = SimpleDocTemplate(pdf_file_path, pagesize=letter)

    # Define styles for different headings and content
    styles = {
        'Heading1': ParagraphStyle(name='Heading1', fontName='DejaVuSansCondensed', fontSize=16, spaceAfter=16, spaceBefore=16, bold=True),
        'Heading2': ParagraphStyle(name='Heading2', fontName='DejaVuSansCondensed', fontSize=14, spaceAfter=14, spaceBefore=14,bold=True),
        'Heading3': ParagraphStyle(name='Heading3', fontName='DejaVuSansCondensed', fontSize=12, spaceAfter=12, spaceBefore=12,bold=True),
        'Normal': ParagraphStyle(name='Normal', fontName='DejaVuSansCondensed', fontSize=10, spaceAfter=8, spaceBefore=8),
        'URL': ParagraphStyle(name='URL', textColor=colors.blue, underline=True, spaceAfter=8),
    }

    # Build the PDF document
    content = [
        Paragraph("SLIDESTER", styles['Heading1']),
        # Image('logo/apple-icon.png', width=140, height=237),
        Spacer(1, 0.01*inch),
        Paragraph("Disclaimer: This content is generated by AI.", styles['Heading3']),
        Spacer(1, 0.5*inch),
        Paragraph("Topic: "+main_topic, styles['Heading1']),
        Spacer(1, 0.05*inch),
        Paragraph("Brief Explanation:", styles['Heading2']),
    ]

    for key,value in notes['brief_explanation'].items():
        content.append(Paragraph(key, styles['Heading3']))
        content.append(Paragraph(value, styles['Normal']))
        content.append(Spacer(1, 0.05*inch))

    content.append(Spacer(1, 0.5*inch))
    content.append(Paragraph("Jargons in the Presentation:- ", styles['Heading2']))

    for key,value in notes['jargons_meaning'].items():
        content.append(Paragraph(key, styles['Heading3']))
        content.append(Paragraph(value, styles['Normal']))
        content.append(Spacer(1, 0.05*inch))

    content.append(Spacer(1, 0.5*inch))
    content.append(Paragraph("Concept Overview:- ", styles['Heading2']))

    for key,value in notes['concepts_explanation'].items():
        content.append(Paragraph(key, styles['Heading3']))
        content.append(Paragraph(value, styles['Normal']))
        content.append(Spacer(1, 0.05*inch))

    content.append(Spacer(1, 0.5*inch))
    content.append(Paragraph("Examples:- ", styles['Heading2']))

    for i in notes['examples']:
        formatted_text = f"<seq/>) {i}"
        content.append(Paragraph(formatted_text, styles['Normal']))

    pdf.build(content, onFirstPage=add_page_number, onLaterPages=add_page_number)

def add_page_number(canvas, docs):
    # Add page numbers
    page_num = canvas.getPageNumber()
    text = "Page %d" % page_num
    canvas.drawRightString(200*mm, 20*mm, text)