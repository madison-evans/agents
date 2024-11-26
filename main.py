import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage
from agents.agent_factory import AgentFactory

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the AgentFactory
factory = AgentFactory(OPENAI_API_KEY=OPENAI_API_KEY)

# Select the agent type
agent_type = input("Enter agent type (conversation_agent or web_search_agent): ").strip()
try:
    agent = factory.factory(agent_type)
except ValueError as e:
    print(e)
    exit()

# Start interacting with the agent
print("Start interacting with the agent (type 'exit' to stop):")
chat_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Add user's input to chat history
    chat_history.append({"role": "human", "content": user_input})

    # Run the agent with the input and chat history
    response = agent.run(user_input, chat_history)

    # Save AI's response to chat history
    ai_response = response["output"]
    chat_history.append({"role": "ai", "content": ai_response})

    # Print the AI's response
    print(f"AI: {ai_response}")
