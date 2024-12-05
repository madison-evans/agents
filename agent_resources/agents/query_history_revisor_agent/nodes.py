from typing import Annotated, TypedDict
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()

class State(TypedDict): 
    messages: Annotated[list, add_messages]

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def general_response(state: State) -> State:
    """
    Handles general conversation flow by generating a response using the LLM.
    """
    # Combine all messages into a single conversation history string
    conversation_history = ""
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            conversation_history += f"User: {message.content}\n"
        elif isinstance(message, AIMessage):
            conversation_history += f"Assistant: {message.content}\n"

    # Create a multi-turn prompt template
    prompt = PromptTemplate(
        input_variables=["conversation_history", "new_user_message"],
        template=(
            "You are a helpful and friendly assistant. Here's the conversation so far:\n"
            "{conversation_history}\n\n"
            "Now the user says:\n{new_user_message}\n\n"
            "Assistant response:"
        ),
    )

    print("current state:")
    print(state["messages"])

    # Format the prompt using the conversation history and the latest user message
    user_message = state["messages"][-1].content
    formatted_prompt = prompt.format(
        conversation_history=conversation_history, 
        new_user_message=user_message
    )

    # Generate the AI response
    ai_message = llm.invoke([HumanMessage(content=formatted_prompt)])

    # Append the AI response to the state
    state["messages"].append(AIMessage(content=ai_message.content))

    return state
