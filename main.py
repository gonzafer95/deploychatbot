from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from chromedriver_py import binary_path
from bs4 import BeautifulSoup
import time
from dotenv import load_dotenv


load_dotenv()

def get_html(url):
  chrome_options = Options()
  chrome_options.add_argument("--headless")
  chrome_options.add_argument("--lang=es")

  #svc = webdriver.ChromeService(executable_path=binary_path)
  driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

  try:
    driver.get(url)
    time.sleep(5)
    html = driver.page_source
    return html
  finally:
    driver.quit()

htmlServices = get_html("https://promptior.ai")
htmlAbout = get_html("https://promptior.ai/about")

soupServices = BeautifulSoup(htmlServices, 'html.parser')
service_titles = soupServices.find_all(class_='service-title')
service_texts = soupServices.find_all(class_='service-text')
title_content = [service_title.get_text() for service_title in service_titles]
text_content = [service_text.get_text() for service_text in service_texts]
services = ['-Servicio= ' + title + ': ' + text + ' ' for title, text in zip(title_content, text_content)]
services_string = ''.join(map(str, services))

soupAbout = BeautifulSoup(htmlAbout, 'html.parser')
about_texts = soupAbout.find_all(class_='text-section')
text_content_about = [about_texts.get_text() for about_texts in about_texts]
about_string = ''.join(map(str, text_content_about))


docs =  [Document(page_content=services_string, metadata={"source": "local"}), Document(page_content=about_string, metadata={"source": "local"})]

embeddings = OpenAIEmbeddings()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
  retriever=vector.as_retriever(),
)
