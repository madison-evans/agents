import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from agents.agent_factory import AgentFactory
from agents.base_agent import Agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

EXIT_COMMAND = 'exit'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

shared_memory = MemorySaver() # shared memory using LangGraph's MemorySaver for conversation persistence

agent_factory = AgentFactory(llm=llm, memory=shared_memory)

agent = agent_factory.factory('web_search_agent') # Choose agent type (i/e 'web_search_agent') and get agent

def chatbot_loop(agent: Agent):
    print("Welcome to the Chatbot! Type 'exit' to end the conversation.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == EXIT_COMMAND:
            print("Goodbye!")
            break

        try:
            ai_message = agent.run(HumanMessage(content=user_input))
            print(f"\n-----\n\nBot: {ai_message.content}\n")
        except Exception as e:
            logging.error("Error generating response", exc_info=True)


if __name__ == "__main__":
    chatbot_loop(agent)