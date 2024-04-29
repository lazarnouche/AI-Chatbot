
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain

class QA_Chain:  
    def __init__(self, texts, embeddings, llm):
        # self.embeddings = texts
        # self.embeddings = embeddings
        self.llm = llm
        self.vectordb = FAISS.from_documents(texts, embeddings)

    def get_qa_chain(self,n_chunks:int) -> ConversationalRetrievalChain:
        
        retriever = self.vectordb.as_retriever(search_kwargs = {"k": n_chunks, "search_type" : "similarity"})

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever = retriever, 
            return_source_documents = True)


        return qa_chain
    
    def get_qa_chain_depreciated(self,prompt_template:str,n_chunks:int) -> RetrievalQA:

        retriever = self.vectordb.as_retriever(search_kwargs = {"k": n_chunks, "search_type" : "similarity"})
        
        qa_chain = RetrievalQA.from_chain_type(
            llm = self.llm,
            chain_type = "stuff", # map_reduce, map_rerank, stuff, refine
            retriever = retriever, 
            chain_type_kwargs = {"prompt": prompt_template},
            return_source_documents = True,
            verbose = False
        )

        return qa_chain
