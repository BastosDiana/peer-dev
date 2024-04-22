import os
from jira import JIRA
from langchain_openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
import numpy as np
import faiss

# Environment variables
os.environ["JIRA_API_TOKEN"] = ""
os.environ["JIRA_USERNAME"] = ""
os.environ["JIRA_INSTANCE_URL"] = ""

# Initialize JIRA client
jira = JIRA(basic_auth=(os.environ["JIRA_USERNAME"], os.environ["JIRA_API_TOKEN"]),
            server=os.environ["JIRA_INSTANCE_URL"])

# Initialize OpenAI for language processing
llm = OpenAI(api_key="", temperature=0, model="gpt-3.5-turbo-instruct")

# Load data from JIRA and split the text data into chunks and then process them.
# How to get the open issues?
# Are this issues the open issues
issues = jira.search_issues("project=MP")

# Store issues along with their texts for reference later
issue_data = []
for issue in issues:
    summary = issue.fields.summary  # Assuming summary is short, doesn't need splitting.
    description = issue.fields.description if issue.fields.description else ""
    # Create a list of tuples (text, key, summary)
    issue_data.append((summary, issue.key, summary))  # Append summary with key and summary for mapping
    for line in description.split('\n'):
        if line.strip():  # Only consider non-empty lines
            issue_data.append((line, issue.key, summary))  # Append description parts with key and summary for mapping

# Extract just the texts for embedding
texts = [data[0] for data in issue_data]  # Get only the texts

# Initialize Langchain OpenAI Embeddings:
embeddings = OpenAIEmbeddings()

# Initialize and populate the FAISS index
db = FAISS.from_documents(texts, embeddings)


# Define the prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a JIRA Assistant. User will ask you for jira cards that you should retrieve."
               "You should return the jira cards to the user related to that topic/input."
               "You shouldn't return jira cards not related to the input."),
    ("system", "Welcome to the Jira Assistant! How can I assist you today?"),
    ("user", "{input}"),
    ("system", "Let me check Jira for tickets related to '{input}'..."),
    MessagesPlaceholder(variable_name="jira_tickets"),
    ("system", "Here are the tickets I found related to '{input}':"),
    MessagesPlaceholder(variable_name="relevant_tickets"),
    ("system", "I couldn't find any tickets related to '{input}'."),
    ("user", "Okay, thank you for checking."),
    ("system", "You're welcome! If you have any more questions or need further assistance, feel free to ask."),
])

# Define the LLM chain with prompt llm and output parser
output_parser = StrOutputParser()
chain = prompt_template | llm | output_parser

# Retrieval chain using the FAISS vector store
# retrieval_chain = FaissRetrieval(index, issue_data)

# Querying the vector store
query = "Find all tickets related to Contract"
docs = db.similarity_search(query)

# Converting the vector store into a retriever
retriever = db.as_retriever()

# Create a prompt for generating the search query
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Please input your query or question related to Jira tickets. "
             "For example, you can ask about the status of a ticket, search for tickets assigned to a specific user, "
             "or inquire about upcoming deadlines.")

])

# Create the history-aware retrieval chain
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# Define the prompt template for continuing conversation with retrieved documents
document_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Here are the Jira tickets related to your query:"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("system", "Please find below the details of the Jira tickets related to your query:"),
    MessagesPlaceholder(variable_name="jira_tickets"),
    ("system", "I couldn't find any Jira tickets related to your query."),
    ("user", "Thank you for checking."),
    ("system", "You're welcome! If you have any more questions or need further assistance, feel free to ask."),
])

# Create a chain to handle retrieved documents
document_chain = create_stuff_documents_chain(llm, document_prompt_template)

# Create a retrieval chain incorporating both the conversation-aware retrieval and document chain
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)


# Function to invoke the retrieval chain with a query
def retrieve_and_answer(query, chat_history):
    response = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": query
    })
    return response


# Example usage: Invoking the retrieval chain with a query
query = "Find all tickets related with contract"
chat_history = [HumanMessage(content="Exist any ticket related with contract?"),
                AIMessage(content="Yes, there are many tickets on board related to contract")]
response = retrieve_and_answer(query, chat_history)

# Print the response
print(response)