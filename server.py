from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve API key from .env
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the ChatGroq model
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# 1. Define the system template for translation
system_template = "Translate the following into {language}"

# 2. Create prompt template using from_messages
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# 3. Output parser for converting model output into a string
parser = StrOutputParser()

# 4. Create the processing chain
chain = prompt_template | model | parser

# Define the FastAPI app
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server using Langchain runnable interfaces"
)

# Add chain routes to the FastAPI app
add_routes(
    app,
    chain,
    path="/chain"
)

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
