import logging
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from .nodes import State, general_response
from agent_resources.base_agent import Agent
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

class QueryHistoryRevisorAgent(Agent): 

    def __init__(self, llm, memory): 
        self.llm = llm 
        self.memory = memory
        self.agent = self.compile_graph()
        self.chat_histories = {}  # A dictionary to track chat histories by session ID

    def compile_graph(self): 
        # Create the state graph
        workflow = StateGraph(State)

        # Add nodes to the graph
        workflow.add_node("start_node", lambda state: state)
        workflow.add_node("general_response", general_response)
        
        # Set entry point
        workflow.add_edge(START, "start_node")
        workflow.add_edge("start_node", "general_response")
        workflow.add_edge("general_response", END)

        # Compile the graph with MemorySaver as the checkpointer
        # This allows LangGraph to handle loading/saving conversation
        # automatically every time we invoke the graph with the same thread_id.
        agent = workflow.compile(checkpointer=self.memory)

        return agent 
    
    def run(self, message: BaseMessage):
        """
        Process a message, update history automatically via MemorySaver, 
        and print the conversation history in the nodes.
        """
        try:
            # Use a fixed thread_id or dynamically assign one if you handle multiple sessions
            thread_id = "default"  
            config = {"configurable": {"thread_id": thread_id}}

            # Invoke the graph with the new incoming message.
            # MemorySaver will automatically load previous state for this thread_id.
            response = self.agent.invoke({"messages": [message]}, config=config)

            ai_message = response["messages"][-1]
            if isinstance(ai_message, AIMessage):
                return ai_message
            else:
                logger.error("Unexpected message type in response.")
                raise ValueError("Expected AIMessage in the response.")
            
        except Exception as e:
            logger.error("Error generating response", exc_info=True)
            return AIMessage(content="Sorry, I encountered an error while processing your request.")
