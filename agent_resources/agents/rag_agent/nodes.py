from typing import Annotated, List, TypedDict
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.tools import BaseTool
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()

class State(TypedDict): 
    messages: Annotated[list, add_messages]
    revised_query: str

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def is_chemistry_related(message: str) -> bool:
    """
    Simple function to determine if a query is about chemistry.
    """
    keywords = ["chemistry", "chemical", "compound", "reaction", "molecule"]
    return any(keyword in message.lower() for keyword in keywords)


def check_query_type(state: State) -> str:
    """
    Conditional function to decide the next node.
    """
    user_message = state["messages"][-1].content
    if is_chemistry_related(user_message):
        return "rewrite_query"
    return "general_response"

def general_response(state: State) -> State:
    """
    Handles general conversation flow by generating a response using the LLM.
    """

    conversation_history = ""
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            conversation_history += f"User: {message.content}\n"
        elif isinstance(message, AIMessage):
            conversation_history += f"Assistant: {message.content}\n"

    prompt = PromptTemplate(
        input_variables=["conversation_history", "new_user_message"],
        template=(
            "You are a helpful and friendly assistant.\n"
            "Conversation history: {conversation_history}\n\n"
            "User: {new_user_message}\n\n"
            "Assistant response:"
        ),
    )

    user_message = state["messages"][-1].content
    formatted_prompt = prompt.format(
        conversation_history=conversation_history, 
        new_user_message=user_message
    )

    ai_message = llm.invoke([HumanMessage(content=formatted_prompt)])

    state["messages"].append(AIMessage(content=ai_message.content))

    return state

def rewrite_query(state: State) -> State: 
    conversation_history = ""
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            conversation_history += f"User: {message.content}\n"
        elif isinstance(message, AIMessage):
            conversation_history += f"Assistant: {message.content}\n"
    prompt = PromptTemplate(
        input_variables=["conversation_history", "new_user_message"],
        template=(
            "Based on the provided 'Conversation history' and 'User query', create a 'Revised Query' which rewords the query to be contextually rich, by including only the relevant elements of the Conversation history\n\n"
            "Conversation history: {conversation_history}\n\n"
            "User Query: {new_user_message}\n\n"
            "Revised Query:"
        ),
    )

    # Format the prompt using the conversation history and the latest user message
    user_message = state["messages"][-1].content
    formatted_prompt = prompt.format(
        conversation_history=conversation_history, 
        new_user_message=user_message
    )
    # Generate the AI response
    ai_message = llm.invoke([HumanMessage(content=formatted_prompt)])
    print(f"revised_query: {ai_message.content}")
    state["revised_query"] = ai_message.content

    return state


def generate_answer_from_retrieval(state: State, retrieve_documents_tool: BaseTool) -> State:
    """
    Generates a final answer using the LLM based on retrieved documents and the revised query.
    """
    # Extract the revised query
    revised_query = state["revised_query"]

    # Retrieve documents using the tool
    retrieved_documents = retrieve_documents_tool.run(revised_query)

    # Combine the content of retrieved documents into a single string
    document_context = "\n".join(
        [
            f"Document {i+1}: {doc.page_content} (source: {doc.metadata['source']})"
            for i, doc in enumerate(retrieved_documents)
        ]
    )

    # Create a prompt template for answer generation
    prompt = PromptTemplate(
        input_variables=["revised_query", "document_context"],
        template=(
            "You are a knowledgeable assistant. Using the provided 'Revised Query' and 'Document Context', "
            "generate a clear and concise answer to the query.\n\n"
            "Revised Query:\n{revised_query}\n\n"
            "Document Context:\n{document_context}\n\n"
            "Final Answer:"
        ),
    )

    formatted_prompt = prompt.format(
        revised_query=revised_query,
        document_context=document_context
    )

    ai_response = llm.invoke([HumanMessage(content=formatted_prompt)])
    state["messages"].append(AIMessage(content=ai_response.content))

    return state