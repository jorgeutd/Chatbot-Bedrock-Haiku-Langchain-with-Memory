# Chatbot with Amazon Bedrock Claude 3 and Memory.
This Chatbot leverages the advanced capabilities of Amazon Bedrock's Claude 3 Haiku model, integrating it with the Retrieval-Augmented Generation (RAG) architecture for dynamic response retrieval. Utilizing FAISS (Facebook AI Similarity Search) for efficient indexing and retrieval of responses, LangChain for memory and chain coversations, and the Bedrock API for access to high-quality generative models, this chatbot represents a sophisticated solution for AI-driven interactions. Built with Streamlit, the chatbot offers an interactive web interface for ease of use.

## Prerequisites
Before you begin, ensure you have the necessary Bedrock IAM permissions and access to the Amazon Bedrock models. If you haven't set this up yet, please refer to the AWS documentation on IAM permissions and accessing Amazon Bedrock Models.


## Getting Started
To get started with the AmazonBedrockGenAI Chatbot, follow these steps:

### Clone the Repository
Clone the project repository to your local machine:

```
https://github.com/jorgeutd/Chatbot-Bedrock-Haiku-Langchain-with-Memory.git
```

### Initial Setup
Open the setup.ipynb notebook in your IDE and execute the setup steps. You may choose to "Run All" or proceed section by section.

### Ingest Documents
Prior to running the application, ingest documents into the vector database using 00_ingest_documets_setup.ipynb to enable the chatbot's retrieval capabilities.

### Run the Application
With the setup complete, launch the chatbot application:

```
streamlit run app.py
```

### Repository Structure
The project is organized as follows:

.
├── 00_ingest_documets_setup.ipynb
├── Data/
├── README.md
├── app.py
├── config.yml
├── logo.png
├── requirements.txt
└── vectorstore/


### License
This project is licensed under the MIT License. Acknowledgments to AWS, Bedrock, and the developers of the utilized technologies for their foundational work.