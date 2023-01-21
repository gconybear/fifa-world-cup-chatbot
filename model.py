import streamlit as st 
import pandas as pd 
import numpy as np  

# import * idc 
from chat_functions import * 


class Bot: 

    def __init__(self, data, embeddings): 

        self.data = data 
        self.embeddings = embeddings

    def generate_response(self, query, show_prompt=False): 
        
        r = answer_query_with_context(query, self.data, self.embeddings, show_prompt=show_prompt) 

        return r
