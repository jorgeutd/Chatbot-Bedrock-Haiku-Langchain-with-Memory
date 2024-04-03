import streamlit_authenticator as stauth
import streamlit as st

import yaml
from yaml.loader import SafeLoader

from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import BedrockEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from PIL import Image
import boto3
from langchain_community.llms import Bedrock
from langchain_community.chat_models import BedrockChat
from typing import Any, Dict, Optional
from botocore.client import Config



# Initialize the boto3 session and BedrocK runtime client
session = boto3.Session(profile_name=, region_name='us-east-1') ## add your profile

bedrock_config = Config(connect_timeout=180, read_timeout=180, retries={'max_attempts': 5})
bedrock_client = session.client('bedrock-runtime')


## Embeddings Model
model_id = "amazon.titan-embed-text-v1"
be = BedrockEmbeddings(
    model_id=model_id,
    credentials_profile_name='', region_name="us-east-1" ## add your profile
    
)

## Load Vector Store

faiss_index = FAISS.load_local('vectorstore', be, allow_dangerous_deserialization=True)
prompt_template = """You are AWS Support AI, a knowledgeable AI assistant specializing in AWS services and solutions. Your primary role is to assist AWS employees by providing accurate and concise guidance on AWS offerings, such as Amazon Aurora, Amazon Bedrock, and other AWS managed services, strictly based on the context provided below. Follow these instructions when answering questions:

- Use English for all responses and maintain a professional tone.
- Begin your answers with "Based on the context provided: ".
- Provide clear, detailed answers using bullet points, focusing solely on the information related to AWS services and solutions within the provided context. If the context does not contain the necessary information to answer a question, respond with, "Sorry, I didn't understand that. Could you rephrase your question?"
- Summarize the key points at the end of your response for clarity and retention.
- Recognize conversational cues indicating the end of an interaction and respond appropriately without seeking additional context. For example, if a user says "Thank you bye," acknowledge the closure of the conversation in a polite and friendly manner.
- Your responses should be informed, precise, and strictly confined to the realm of AWS documentation and best practices, as outlined in the context provided.

Your expertise in AWS services positions you as a crucial resource for AWS employees seeking guidance. Ensure your responses are as informative and accurate as possible, adhering strictly to the provided context.

<context>{context}</context>

Begin:
Question: {question}

think step by step and answer the question with utmost precision and accuracy.

Assistant:"""

_template = """Human: Given the following conversation and a follow up question, rephrase the follow up question to be a 
standalone question without changing the content in given question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question: Assistant:"""

condense_question_prompt_template = PromptTemplate.from_template(_template)
qa_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

## AWS bedrock LLM
bedrock_model_id = "anthropic.claude-3-haiku-20240307-v1:0"  # Bedrock model_id

model_kwargs_claude = {
    "temperature": 0.0,
    "top_p": 0.99,
    "max_tokens": 3000,
    "stop_sequences": ["\n\nHuman"]
}


llm =  BedrockChat(model_id=bedrock_model_id, 
                   client=bedrock_client,
                   model_kwargs=model_kwargs_claude)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
question_generator = LLMChain(llm=llm, prompt=condense_question_prompt_template, memory=memory)
doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)
qa_chain = ConversationalRetrievalChain(
    retriever=faiss_index.as_retriever(search_type="similarity", search_kwargs={'k': 5}),
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
    memory=memory,
)

### new session rest

def reset_chat():
    st.session_state.messages = [
        {"role": "user", "content": 'Hello!'},
        {"role": "system", "content": "Hello! My name is AWS Support AI and I'm happy to assist you with providing guidance in any AWS services. As an AI assistant, I specialize in providing accurate and concise answers related to AWS. How can I help you today?"}
    ]
    st.session_state.chat_history = []
    qa_chain.memory.clear()

image = Image.open('./logo.png')
with st.sidebar:
    st.image(image, width=300)
    welcome_placeholder = st.empty()
    
    if st.button("Reset Chat"):
        reset_chat()


with open('config.yml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Replace this line
#name, authentication_status, username = authenticator.login('Login', 'main')

# With this line
name, authentication_status, username = authenticator.login(fields=['username', 'name'])

if authentication_status:
    with st.sidebar:
        authenticator.logout('Logout', 'main')
    welcome_placeholder.write(f'Welcome *{name}*')
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')

if authentication_status:
    # Initialize chat history
    if "messages" not in st.session_state:
        reset_chat()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if question := st.chat_input('How can I help you?'):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner('Just a moment, I am looking for an answer for you...'):
                result = qa_chain.invoke({'question': question, 'chat_history': st.session_state.chat_history})
                full_response = result['answer']
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.session_state.chat_history.append((question, full_response))