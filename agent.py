import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langsmith import traceable
from langchain_community.tools.shell.tool import ShellTool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts.prompt import PromptTemplate
import subprocess
import requests
from typing import Optional, Dict, Any

@tool
def get_github_issue_details(issue_number: int) -> Dict[str, Any]:
    """
    Fetches details of a specific issue from the 'trilogy-group/eng-maintenance' GitHub repository.

    Args:
        issue_number (int): The number of the issue.

    Returns:
        dict: A dictionary containing issue details.

    Raises:
        requests.exceptions.RequestException: If the request fails for any reason.
    """
    owner = "trilogy-group"
    repo = "eng-maintenance"
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
    headers = {}

    headers['Accept'] = 'application/vnd.github+json'
    access_token = os.environ.get('GITHUB_TOKEN')
    if access_token:
        headers['Authorization'] = f'token {access_token}'
    else:
        raise ValueError("GITHUB_TOKEN environment variable is not set.")

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

@tool
def evaluate_response(question: str) -> str:
    """
    Evaluates the answer to a question.

    Args:
        question (str): The question to evaluate.

    Returns:
        str: The evaluation of the answer.
    """

    prompt = (
        "Please review the question and generate response in markdown format (to be added as a github issue comment):\n"
        f"{question}\n"
    )

    # Note: we must use the same embedding model that we used when uploading the docs
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Querying the vector database for "relevant" docs
    document_vectorstore = PineconeVectorStore(index_name="lithium-manuals", embedding=embeddings)
    retriever = document_vectorstore.as_retriever()
    context = retriever.get_relevant_documents(prompt)

    # Adding context to our prompt
    template = PromptTemplate(template="{query} Context: {context}", input_variables=["query", "context"])
    prompt_with_context = template.invoke({"query": prompt, "context": context})

    # Asking the LLM for a response from our prompt with the provided context
    results = llm.invoke(prompt_with_context)
    return results.content

@tool
def create_github_issue_comment(issue_number: int, content: str) -> str:
    """
    Creates a new comment on a specific issue in the 'trilogy-group/eng-maintenance' GitHub repository.

    Args:
        issue_number (int): The number of the issue.
        content (str): The content of the comment.

    Returns:
        str: A success or error message.
    """
    owner = "trilogy-group"
    repo = "eng-maintenance"
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
    headers = {}

    headers['Accept'] = 'application/vnd.github.v3+json'
    access_token = os.environ.get('GITHUB_TOKEN')
    if access_token:
        headers['Authorization'] = f'token {access_token}'
    else:
        return "GITHUB_TOKEN environment variable is not set."

    contentStarting = "This is a comment from a development AI agent:"
    content = contentStarting + "\n" + content
    data = {"body": content}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        return "Comment created successfully."
    else:
        return f"Failed to create comment. Status code: {response.status_code}"



# List of tools to use
tools = [
    ShellTool(ask_human_input=True),
    get_github_issue_details,
    evaluate_response,
    create_github_issue_comment,
    # Add more tools if needed
]


# Configure the language model
llm = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0)


# Set up the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert chatbot that can answer technical queries. You grab the github issue and filter out the question from it. You then evaluate the answer and post the answer in a comment in the same issue.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# Bind the tools to the language model
llm_with_tools = llm.bind_tools(tools)


agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Main loop to prompt the user
while True:
    user_prompt = input("Prompt: ")
    list(agent_executor.stream({"input": user_prompt}))
