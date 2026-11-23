from dotenv import load_dotenv             # Imports the function used to load variables from the .env file
import os                                  # os to read environment variables
from langchain_openai import ChatOpenAI    # Imports the OpenAI-compatible chat model from LangChain.

load_dotenv()          # load the variables from the .env file



llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"), 
    temperature=0)   
# Creates the LLM.
# The model name is read from OPENAI_MODEL in the .env file.
 