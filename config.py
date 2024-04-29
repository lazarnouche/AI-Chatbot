PERSIST_DIR = "vectorstore"  # replace with the directory where you want to store the vectorstore
LOGS_FILE = "logs/log.log"  # replace with the path where you want to store the log file

CHATBOT = "Fusion4"

CONF = {
        "Etch": {
            "name": "Semiconductor Etching Lithography and Metrology Virtual Assistant",
            "logo": "etch.png",
            "file": "doc/AMAT/AMAT_PART_I_TO_II.pkl",
            "modules" : ["pages/2_🐾_Etch_Profiles.py",
                         "pages/3_📈_SEM_Detection.py"]
        },
        "Fusion4": {
            "name": "Mega Industry Control Systems Virtual Assistant",
            "logo": "mega.png",
            "file": "doc/Fusion4/Rev02_Fusion4_Communication_Manual.pkl",
            "modules" : ["pages/1_FAQ.py"]
        },

}

           

FILE_DIR = "doc/"
MODULES = {"Etch" : 
           {"recipe_path" :"doc/AMAT/Materials_gas_systems_etching.xlsx",
            "image_dir" :"doc/AMAT//images"},
            "SEM" : 
            {"model_path": "doc/models/model_EPOCH_19_feb_29.torch",
                     "image_dir" :"doc/AMAT//images"}
            
          }

NEG_ANSWER = "I'm sorry, I don't have enough information to answer your question."
prompt_template = f"""Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. Don't try to make up an answer, if you don't know just say **{NEG_ANSWER}**
2. Answer in the same language the question was asked.

{{context}}

Question: {{question}}
Answer:"""
k = 5  # number of chunks to consider when generating answer
NUM_IMAGES = 2
MAKE_CHUNKS = True