import os
# os.chdir(os.path.abspath(".."))
import re
import cv2
import numpy as np
import config
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from modules.module import Module_Class
from util.preprocess import read_json
from util.signal_utils import sampler

nltk.download("punkt")

class Etch(Module_Class):
    def __init__(self,module=config.MODULES):
        super().__init__(module)

        self.df_lay = pd.DataFrame()
        self.df_rec= pd.DataFrame()
        self.dic_results = {}
        self.unities = None
        self.variables = []
        self.load_recipes()

    def load_recipes(self):

        if not os.path.exists(self.paths["recipe_path"]):
            raise FileNotFoundError(f"""Cannot find the file {self.paths["recipe_path"]}""")
        path = self.paths["recipe_path"]
        self.df_lay = pd.read_excel(path, "layers")
        self.df_rec = pd.read_excel(path, "recipes")
        self.df_lay = self.df_lay.fillna("")
        self.df_rec = self.df_rec.fillna("")
        self.df_lay.layer_ID = self.df_lay.layer_ID.astype(int)
        self.df_rec.layer_ID = self.df_rec.layer_ID.astype(int)
        self.df_lay = self.df_lay.astype(str)
        self.df_rec = self.df_rec.astype(str)
        self.df_lay["layer_ID"] = self.df_lay.layer_ID.astype(int)
        self.df_rec[["layer_ID",
                "recipe_ID"]] = self.df_rec[["layer_ID",
                                                   "recipe_ID"]].astype(int)
        self.unities = self.df_rec.loc[self.df_rec.recipe_ID == -1]
        self.df_lay = Etch.add_queries_to_df_layer(self.df_lay)
        self.LOGGER.info(f"Sucesfully load recipes and layers.")

    
    @staticmethod
    def create_query(row,query):

        if row["Etch"].split()[0].lower() == 'no':
            return query.format("Cure", row["Layer"], row["Etch Chemistry"])
        query = query.format("Etch", row["Layer"], row["Etch Chemistry"])
        if row["Cure"] == '':
            return query
        if row["Cure"].split()[0].lower() != 'no':
            query+= f"""" (followed by {row["Cure"]} cure )"""

        return query
    
    @staticmethod
    def add_queries_to_df_layer(df_lay):
          
        queries = []
        query = "How to {} {} layer with {} chemistery?"
        for index,row in df_lay.iterrows():
            queries.append(Etch.create_query(row,query))
        df_lay["Query"] = queries

        return df_lay



    @staticmethod
    def parse_query(query, df_lay):
        unique_key = "55ea7882-3ea3-4733-9bd0-91c13c1f6e60"
        chem_list = [
            (lay_id, chem.split()[0])
            for (lay_id, chem) in df_lay[["layer_ID", "Etch Chemistry"]].values
            if chem
        ]
        chem_list = [(chem[0], chem[1].split("/")) for chem in chem_list]
        look_up_recipe = []
        new_query = query
        for id, chem in chem_list:
            dic = {}
            for i, gas in enumerate(chem):
                pattern = re.compile(f"\\b{re.escape(gas)}\\b", flags=re.IGNORECASE)
                pat_search = pattern.search(query)
                if bool(pat_search):
                    dic[gas] = True
                    new_query = pattern.sub(unique_key, new_query)
                else:
                    dic[gas] = False
            if all(dic.values()):
                dic["id"] = id
                look_up_recipe.append(dic)
        if look_up_recipe == []:
            raise ValueError(f"""Cannot find the question " {str(query)} " within my available information""")
        replaced_sentence = "layer_ID equal to "
        for i, recipe in enumerate(look_up_recipe):
            replaced_sentence += str(recipe["id"])
            if i < len(look_up_recipe) - 1:
                replaced_sentence += " and "
        new_query_list = nltk.word_tokenize(new_query)
        nq_index = [i for (i, word) in enumerate(new_query_list) if unique_key not in word]
        nq_n_index = [i for (i, word) in enumerate(new_query_list) if unique_key in word]
        nq_index.append(nq_n_index[0])
        new_query_list = [new_query_list[i] for i in nq_index]
        for i, query_word in enumerate(new_query_list):
            if unique_key in query_word:
                new_query_list[i] = replaced_sentence

        return (" ".join(new_query_list), look_up_recipe)
    
    @staticmethod
    def get_recipes(df_rec,ids):
     ids.append(-1)
     return df_rec[df_rec.layer_ID.isin(ids)]

    @staticmethod
    def parse_recipe(df_result):

        # df_unities = df_result.loc[df_result.recipe_ID == -1]
        df_result = df_result.loc[df_result.recipe_ID != -1]
        df_result= df_result.replace('', np.nan)
        if df_result.Path.isnull().values.all():
            raise ValueError(f"""No path has been assigned to this layer ID check layer & recipe files.""")
        df_result = df_result[df_result.columns[df_result.notna().any()].tolist()]
        col = list(df_result.columns)
        col = col[col.index('layer_ID')+1:col.index('Path')]
        col_var = list(df_result[col].columns[df_result[col].nunique() > 1])
        dic_results = {}
        for var in col_var:
            numeric_rows = pd.to_numeric(df_result[var], errors='coerce').notna()
            dic_results[var] = df_result[numeric_rows]

        return dic_results

    @staticmethod
    def create_and_center_pattern(img_path):
    
        img = cv2.imread(img_path)
        if img_path.endswith(".png"):
            end_ = ".png"
        else:
            end_ = ".jpg"
        js_path = img_path.replace(end_,".json")
        js_file = read_json(js_path)
        filled = np.ones_like(img)*230
        cnts,centroid = [],[]
        for coor in js_file['shapes']:
            xy = np.array(coor['points']).astype(np.int32)
            M = cv2.moments(xy)
            # Calculate centroid coordinates
            if M["m00"] != 0:
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])
            else:
                centroid_x, centroid_y = 0, 0
            xy[:,0] = xy[:,0]-centroid_x+int(filled.shape[1]/2)
            xy[:,1] =xy[:,1]-centroid_y+int(filled.shape[0]/2)
            cnts.append(xy)
            centroid.append([centroid_x, centroid_y])
        centroid = np.array(centroid)
        sorted_index = np.argsort(centroid[:,0])
        centroid = centroid[sorted_index]
        interp_cnts = np.array(sampler(cnts, 200))
        interp_cnts_mean = np.mean(interp_cnts,axis=0).astype(int)
        diff_x = np.mean(np.diff(centroid[:,0]))
        nb_pattern = int(img.shape[1]/diff_x)*2
        
        if nb_pattern%2 ==0:
            nb_pattern +=1
        rg = int(nb_pattern/2)*diff_x
        list_x_offset = np.linspace(-rg,rg,nb_pattern,dtype=int)
        list_x_offset =list_x_offset.astype(int)
        for x_offset in list_x_offset:
            new_cnts = np.array([interp_cnts_mean[:,0]+x_offset,interp_cnts_mean[:,1]]).T
            filled = cv2.fillPoly(filled, pts = [new_cnts], color =(100,102,255),
                                 lineType=cv2.LINE_AA)
        
            #crop image
        interp_min = interp_cnts_mean[:,1].min()-interp_cnts_mean[:,1].min()*0.1
        interp_max = interp_cnts_mean[:,1].max()+interp_cnts_mean[:,1].max()*0.1
        
        lower_pos = int(interp_min) if interp_min>0 else 0
        higher_pos = int(interp_max) if interp_max<filled.shape[0] else filled.shape[0]
        filled = filled[lower_pos:higher_pos]

        return filled
  
    def diplay_mask(self,var):
        
        text = str(var + ": {}" +self.unities.loc[0,var]) 
        self.LOGGER.info(f"Display profile based on: {text}.")
        figs = {}
        # bbox = dict(boxstyle="round", fc="0.9")
        df = self.dic_results[var]
        for index,row in df.iterrows():

            fig, ax = plt.subplots()
            image_path = self.paths["image_dir"]+os.sep+row.Path.split()[0]
            filled  = Etch.create_and_center_pattern(image_path)
            ax.imshow(filled)
            #text = str(var + ": "+ str(row[var]) + ' '+self.unities.loc[0,var]) 

            # ax.annotate(text,xy=(int(filled.shape[0]/(2)),10),fontsize=12) 
            ax.axis("off")
            figs[row[var]] = fig
        
        # Log a message indicating that the function has finished and return the answer.
        self.LOGGER.info(f"Profile display is complete.")
            
        return figs, text

    def handle_actions(self,query):

        # Make sure self.load_recipes() has been called before running this method
        self.LOGGER.info(f"Start answering based on prompt: {query}.")
        _, recipes = Etch.parse_query(query,self.df_lay)
        ids = [rec['id'] for rec in recipes]
        df_result = Etch.get_recipes(self.df_rec,ids)
        if not os.path.exists(self.paths["image_dir"]):
            FileNotFoundError(f"""Cannot find the directory {self.paths["image_dir"]}""")

        self.dic_results = Etch.parse_recipe(df_result)
        self.variables = list(self.dic_results.keys())
         

if __name__ == "__main__":
    query = 'SF6/CH2F2'
    etch = Etch()
    try:
        etch.handle_actions(query)
        var = etch.variables[0]
        artists = etch.diplay_mask(var)
    except (FileNotFoundError, ValueError) as e:
        print("Error:", e)