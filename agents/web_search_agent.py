from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from .tools import initialize_tools

class WebSearchAgent:
    def __init__(self, OPENAI_API_KEY: str):
        self.llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-0125")
        
        # Initialize tools
        self.tools = initialize_tools()
        
        # Define a prompt for the agent
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Use the Tavily search tool for queries that require external or up-to-date information, "
                    "especially for topics not commonly found in standard encyclopedias. For common knowledge, respond directly without using tools."
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        
        # Create the tool-calling agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def run(self, input_text: str, chat_history: list = None):
        """
        Run the agent with the given input text and optional chat history.
        """
        input_data = {"input": input_text}
        if chat_history:
            input_data["chat_history"] = chat_history

        return self.executor.invoke(input_data)
