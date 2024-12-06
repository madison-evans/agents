import logging
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, AIMessage
from .nodes import State, general_response, check_query_type, rewrite_query, generate_answer_from_retrieval
from agent_resources.base_agent import Agent

logger = logging.getLogger(__name__)

class RAGAgent(Agent): 

    def __init__(self, llm, memory): 
        self.llm = llm 
        self.memory = memory
        self.agent = self.compile_graph()
        self.chat_histories = {}  # A dictionary to track chat histories by session ID

    def compile_graph(self):
        """
        Compile the graph for the RAG pipeline with all necessary nodes and edges.
        """
        # Create the state graph
        workflow = StateGraph(State)

        # Add nodes to the graph
        workflow.add_node("general_response", general_response)
        workflow.add_node("rewrite_query", rewrite_query)
        workflow.add_node("generate_answer_from_retrieval", generate_answer_from_retrieval)

        # Set conditional entry point to decide between RAG or general response
        workflow.add_conditional_edges(
            START,
            check_query_type,
            path_map={
                "general_response": "general_response",
                "rewrite_query": "rewrite_query",
            },
        )

        # Define RAG pipeline flow: rewrite query → generate answer
        workflow.add_edge("rewrite_query", "generate_answer_from_retrieval")

        # End the graph
        workflow.add_edge("general_response", END)
        workflow.add_edge("generate_answer_from_retrieval", END)

        # Compile the graph with MemorySaver as the checkpointer
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
