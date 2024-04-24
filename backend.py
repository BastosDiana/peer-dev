import os
import gradio as gr
from jira import JIRA
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma
from dotenv import load_dotenv


class Document:
    def __init__(self, text, metadata=None, page_content=None):
        self.text = text
        self.metadata = metadata if metadata else {}
        self.page_content = page_content


class JiraCon:
    def __init__(self):
        load_dotenv()

        # Initialize JIRA client
        self.jira = JIRA(basic_auth=(os.environ["JIRA_USERNAME"], os.environ["JIRA_API_TOKEN"]),
                         server=os.environ["JIRA_INSTANCE_URL"])

    def search_issues(self, jql):
        return self.jira.search_issues(jql)


def handle_jira_docs():
    # Load data from JIRA and split the text data into chunks and then process them.
    issues = JiraCon().search_issues("project=MP")

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

    return documents


load_dotenv()
# Initialize OpenAI for language processing
llm = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], temperature=0, model="gpt-3.5-turbo")

# Initialize Langchain OpenAI Embeddings:
embeddings = OpenAIEmbeddings()

# Load the documents from JIRA
documents = handle_jira_docs()

# Initialize and populate the CHROMA DB index
db = Chroma.from_documents(documents, embeddings)

input_word = "contract"
query = f"Find me all tickets related with {input_word}"
docs = db.similarity_search(query)

# Converting the vector store into a retriever
retriever = db.as_retriever()

# Create a prompt for generating the search query
prompt = ChatPromptTemplate.from_messages([
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
    ("system", "Please input your query or question related to Jira tickets. "
               "For example, you can ask about the status of a ticket, search for tickets assigned to a specific user, "
               "or inquire about upcoming deadlines."),
    ("user", "Okay, thank you for checking."),
    ("system", "You're welcome! If you have any more questions or need further assistance, feel free to ask."),

])

# Create the history-aware retrieval chain
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# Define the prompt template for continuing conversation with retrieved documents
document_prompt_template = ChatPromptTemplate.from_messages([
    ("system", f"Here are the Jira tickets related to {input_word}]:\n\n '{{context}}':"),
])
document_prompt_template.invoke({"context": documents})

# Create a chain to handle retrieved documents
document_chain = create_stuff_documents_chain(llm, document_prompt_template)

# Create a retrieval chain incorporating both the conversation-aware retrieval and document chain
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)


# For Frontend - Define the function to query Jira tickets
first_input = True
def query_jira_tickets(input_word):
    global first_input

    if first_input:
        # Check if the input is a greeting or unclear
        if input_word.lower() in ["hello", "hi", "hey", "greetings", "what's up"]:
            first_input = False  # Mark as not first input
            return "Please input your query or question related to Jira tickets. " \
                   "For example, you can ask about the status of a ticket, search for tickets assigned to a specific user, " \
                   "or inquire about upcoming deadlines."
        else:
            first_input = False  # Mark as not first input
            return "Here are the tickets I found related to '{}':".format(input_word)

    # Filter the documents to include only those related to the input word
    relevant_tickets = [doc for doc in documents if input_word.lower() in doc.text.lower()]

    # If no relevant tickets are found, return a message
    if not relevant_tickets:
        return f"No tickets related to '{input_word}' were found."

    # Invoke the retrieval chain
    chat_history = []
    template = {
        "input": input_word,
        "chat_history": chat_history,
        "relevant_tickets": relevant_tickets
    }
    response = retrieval_chain.invoke(template)

    # Convert the response to a format suitable for display
    output = response.get("answer", "No answer found")

    # Prepend the message to the list of tickets
    output_with_message = f"Here are the tickets I found related to '{input_word}':\n{output}"
    return output_with_message


print(gr)
# Create a Gradio interface
iface = gr.Interface(
    fn=query_jira_tickets,
    inputs=gr.Textbox(lines=1, label="Enter a word or phrase:"),
    outputs="text",
    title="Peer AI",
    description="Enter a word or phrase to search for related Jira issues."
)

# Launch the interface
iface.launch()
