# views.py
from django.http import JsonResponse
from .models import QA
from .forms import PromptForm
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.urls import reverse_lazy
from django.contrib.auth import logout
import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, TextStreamer, pipeline
from django.views.decorators.csrf import csrf_exempt

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
loader = PyPDFDirectoryLoader(r"D:\Nasa_Data_Base\nasa dATA")
docs = loader.load()
embeddings = HuggingFaceInstructEmbeddings(
    model_name="BAAI/bge-large-en-v1.5", model_kwargs={"device": DEVICE}
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
texts = text_splitter.split_documents(docs)
db = FAISS.from_documents(texts, embeddings)
model_name_or_path ="TheBloke/Llama-2-7b-Chat-GPTQ"
model_basename = "model"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(
    model_name_or_path,
    revision="gptq-4bit-128g-actorder_True",
    model_basename=model_basename,
    use_safetensors=True,
    trust_remote_code=True,
    inject_fused_attention=False,
    device=DEVICE,
    quantize_config=None,
)



DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Do not generate any harmful, unethical, dangerous or illegal content. Provide the answer according to information in given context and
If a question's answer is not in context or you do not know the answer, respond only with "For this query please contact to admin branch". """.strip()


def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""
[INST] <<SYS>>
{system_prompt}
<</SYS>>

{prompt} [/INST]
""".strip()

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15,
    streamer=streamer,
)
llm = HuggingFacePipeline(pipeline=text_pipeline)
SYSTEM_PROMPT = """ Use the provided context to answer the question according to correct information in the context. If the answer is not present in the context, respond only with "For this query please contact to admin branch".
Do not attempt to speculate or generate any worng response."""
template = generate_prompt(
    """
{context}

Question: {question}
""",
    system_prompt=SYSTEM_PROMPT,
)
prompt = PromptTemplate(template=template, input_variables=["context", "question"])
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

@login_required(login_url=reverse_lazy('custom_login'))  # Protect the view with @login_required
@csrf_exempt
def get_answer(request):
    if request.method == 'POST':
        form = PromptForm(request.POST)
        if form.is_valid():
            question = form.cleaned_data['prompt']
            answer = qa_chain(question)  # Replace with your answer generation logic
            print('answer',answer)
            QA.objects.create(prompt=question, answer=answer['result'])
            return JsonResponse({'answer': answer['result']})  # Return JSON response with the answer
        else:
            return JsonResponse({'error': 'Invalid form data'}, status=400)

    # For GET requests (initial page load or refresh), retrieve and display chat history
    form = PromptForm(initial={'prompt': ''})  # Set the initial value to an empty string
    qas = QA.objects.all().order_by('created_at')

    return render(request, 'index.html', {'form': form, 'qas': qas})

@csrf_exempt
def custom_login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('get_answer')  # Redirect to the get_answer view upon successful login
        else:
            messages.error(request, 'Invalid username or password. Please try again or sign up.')
            # return redirect('signup')  # Redirect to the signup view

    return render(request, 'registration/login.html')


@csrf_exempt
def custom_logout_view(request):
    logout(request)
    return redirect('custom_login')

    # return render(request, 'registration/logout.html')
@csrf_exempt
def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            login(request, user)
            messages.success(request, 'Account created successfully.')
            return redirect('custom_login')
    else:
        form = UserCreationForm()
    return render(request, 'registration/signup.html', {'form': form})