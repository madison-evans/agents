import logging
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from agents.base_agent import Agent
from agents.tools.tool_registry import ToolRegistry



logger = logging.getLogger(__name__)


class WebSearchAgent(Agent):
    """
    LangGraph-based WebSearch agent implementation with MemorySaver for persistence.
    """

    def __init__(self, llm, memory):
        """
        Initialize the WebSearch agent using LangGraph.

        :param llm: The language model.
        :param memory: Persistent memory for managing conversation history.
        """

        # Retrieve the tavily_search tool from the registry
        web_search_tool = ToolRegistry.get_tool('tavily_search', max_results=1)

        self.tools = [web_search_tool]
        self.llm = llm
        self.memory = memory

        # Create a prebuilt React agent with tools and memory
        self.agent = create_react_agent(
            self.llm,
            tools=self.tools,
            checkpointer=self.memory,
        )


    def run(self, message: HumanMessage) -> AIMessage:
        """
        Process a HumanMessage and return an AIMessage response using a single-step invoke.

        :param message: User's input message.
        :return: AIMessage response.
        """
        try:
            thread_id = "default"  # You can customize this for multi-user support
            config = {"configurable": {"thread_id": thread_id}}

            # Use invoke for a single-step interaction
            response = self.agent.invoke({"messages": [message]}, config=config, debug=True)
            ai_message = response["messages"][-1]  # Assuming the last message is the AI response
            if isinstance(ai_message, AIMessage):
                return ai_message
            else:
                logger.error("Unexpected message type in response.")
                raise ValueError("Expected AIMessage in the response.")

        except Exception as e:
            logger.error("Error generating response", exc_info=True)
            return AIMessage(content="Sorry, I encountered an error while processing your request.")
