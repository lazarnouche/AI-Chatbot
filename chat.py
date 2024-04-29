from typing import List, Any
import config
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
import logging
from api_auth.api_key import OPENAI_API_KEY
from util.preprocess import load_file, read_images_in_pdf
from util.qa_chain import QA_Chain

class QA_Chat:
    def __init__(self):
        logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(config.LOGS_FILE),
            logging.StreamHandler(),
        ],
        )
        self.LOGGER = logging.getLogger(__name__)
        self.chatbot = config.CHATBOT
        self.process_document_and_qa_chain()

    def process_document_and_qa_chain(self):
        
        filepath = config.CONF[self.chatbot]['file']
        docs = load_file(filepath)
        
        # if config.MAKE_CHUNKS:
        #     docs = split_into_chuncks(docs,chunk_size=3000)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        # Split the documents into chunks of size 1000 using
        # document = text_splitter.create_documents(docs)
        texts = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # docsearch = Chroma.from_documents(texts, embeddings)
        llm=OpenAI(
            openai_api_key = OPENAI_API_KEY,
            model_name="gpt-3.5-turbo-instruct",
            temperature=0.2,
            max_tokens=300,
        )
        self.QAC = QA_Chain(texts,embeddings,llm)

    def answer(self,prompt: str, chat_history: List[tuple[str,Any]] = []) -> str:
        
        # Log a message indicating that the function has started
        self.LOGGER.info(f"Start answering based on prompt: {prompt}.")
        prompt_template = PromptTemplate(template=config.prompt_template, input_variables=["context", "question"])
        # Create a prompt template using a template from the config module and input variables
        # representing the context and question.
        

        qa =  self.QAC.get_qa_chain(config.k)
        
        # Log a message indicating the number of chunks to be considered when answering the user's query.
        self.LOGGER.info(f"The top {config.k} chunks are considered to answer the user's query.")
        
        # Call the VectorDBQA object to generate an answer to the prompt.
        try:
            # result = qa.invoke(prompt)
            result = qa({"question":prompt,"chat_history":chat_history})
            
        except Exception as er:
            result = {}
            result["answer"] = str(er)
            result["source_documents"] = ''


        answer = result["answer"]
        pages = []
        for res in result['source_documents']:
            pages.append(dict(res)['metadata'])
            print(f"{res}\n")
        # print(f"\n")
        # pages = list(set(pages))
        # Log a message indicating the answer that was generated
        self.LOGGER.info(f"The returned answer is: {answer}")
        
        filepath = config.CONF[self.chatbot]['file']
        
        if config.NEG_ANSWER.strip()== answer.strip(): return answer, []
        
        imgs = read_images_in_pdf(pages,config.NUM_IMAGES)

        if imgs == [] :  self.LOGGER.info(f"Cannot find {filepath.replace('.pkl','.pdf')}")

        # Log a message indicating that the function has finished and return the answer.
        self.LOGGER.info(f"Answering module over.")

        return answer, imgs
    

qa_chat = QA_Chat()



