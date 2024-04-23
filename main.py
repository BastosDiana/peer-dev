import os
from jira import JIRA
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma
import spacy

# Modify the creation of Document instances to include page_content and metadata
# pre-trained English language model from the spaCy library
# nlp object becomes a language processing pipeline
nlp = spacy.load("en_core_web_sm")
class Document:
    def __init__(self, text, metadata=None, page_content=None):
        self.text = text
        self.metadata = metadata if metadata else {}
        self.page_content = page_content


# Access environment variables
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_USERNAME = os.getenv("JIRA_USERNAME")
JIRA_INSTANCE_URL = os.getenv("JIRA_INSTANCE_URL")

# Initialize JIRA client
jira = JIRA(basic_auth=(os.environ["JIRA_USERNAME"], os.environ["JIRA_API_TOKEN"]),
            server=os.environ["JIRA_INSTANCE_URL"])

# Initialize OpenAI for language processing
llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"], temperature=0, model="gpt-3.5-turbo")

# Load data from JIRA and split the text data into chunks and then process them.
issues = jira.search_issues("project=MP")

# Store issues along with their texts for reference later
issue_data = []
for issue in issues:
    summary = issue.fields.summary
    description = issue.fields.description if issue.fields.description else ""

    # Create Document objects for summary and description
    summary_doc = Document(summary, {"issue_key": issue.key, "source": "summary"}, summary)
    issue_data.append((summary_doc, issue.key, summary_doc.text))

    for line in description.split('\n'):
        if line.strip():
            description_doc = Document(line, {"issue_key": issue.key, "source": "description"}, line)
            issue_data.append((description_doc, issue.key, summary_doc.text))

# Extract just the Document objects
documents = [data[0] for data in issue_data]

metadata = [doc.metadata for doc in documents]
page_contents = [doc.page_content for doc in documents]

#print("metadata:",metadata, "page_contents:",page_contents)

# Extract the text content from Document objects
texts = [doc.text for doc in documents]

# Initialize Langchain OpenAI Embeddings:
embeddings = OpenAIEmbeddings()

# Initialize and populate the CHROMA DB index
db = Chroma.from_documents(documents, embeddings)

# query it
query = "Find me all tickets related with contract"
docs = db.similarity_search(query)

# Define the prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a JIRA Assistant. User will ask you for jira cards that you should retrieve."
               "You should return the jira cards to the user related to that topic/input."
               "You shouldn't return jira cards not related to the input."),
    ("system", "Welcome to the Jira Assistant! How can I assist you today?"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("system", "Let me check Jira for tickets related to '{input}'..."),
    MessagesPlaceholder(variable_name="jira_tickets"),
    ("system", "Here are the tickets I found related to '{input}':"),
    MessagesPlaceholder(variable_name="relevant_tickets"),
    ("system", "I couldn't find any tickets related to '{input}'."),
    ("user", "Okay, thank you for checking."),
    ("system", "You're welcome! If you have any more questions or need further assistance, feel free to ask."),
])

# set up the chain that takes a question and the retrieved documents and generates an answer
document_chain = create_stuff_documents_chain(llm, prompt_template)

# Converting the vector store into a retriever
# use the retriever to dynamically select the most relevant documents and pass those in for a given question.
retriever = db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Create a prompt for generating the search query
query_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Please input your query or question related to Jira tickets. "
             "For example, you can ask about the status of a ticket, search for tickets assigned to a specific user, "
             "or inquire about upcoming deadlines.")

])

# Create the history-aware retrieval chain
retriever_chain = create_history_aware_retriever(llm, retriever, query_prompt)

# Define the prompt template for continuing conversation with retrieved documents
document_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Here are the Jira tickets related to '{input}':"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("system", "Please find below the details of the Jira tickets related to '{input}':"),
    MessagesPlaceholder(variable_name="jira_tickets"),
    ("system", "I couldn't find any Jira tickets related to '{input}'."),
    ("user", "Thank you for checking."),
    ("system", "You're welcome! If you have any more questions or need further assistance, feel free to ask."),
])

# Create a chain to handle retrieved documents
document_chain = create_stuff_documents_chain(llm, document_prompt_template)

# Create a retrieval chain incorporating both the conversation-aware retrieval and document chain
document_retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)


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