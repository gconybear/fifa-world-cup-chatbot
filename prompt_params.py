OG_HEADER = """Answer the question as truthfully as possible using the provided context\n\nContext:\n"""  
H1 = """Answer the question as truthfully as possible using the provided context. If the answer isn't in the context but you
think you know it, go ahead and try but you must inform the user. If you don't know the answer, respond 'I'm not sure about this one'\n\nContext:\n"""

H2 = """Answer the question as truthfully as possible using the provided context. If the answer isn't in the context but you
think you know it, go ahead and try but you must inform the user that you're not sure. If you don't know the answer, respond 'I'm not sure about this one'
and then provide them with a recommended google query that should answer their question. \n\nContext:\n"""


prompt_params = {
    "header": OG_HEADER, 
}