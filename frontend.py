import gradio as gr


# Define the function to query Jira tickets
def query_jira_tickets(input_word):
    # Call the backend API endpoint to query Jira tickets
    # Return the response
    pass


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