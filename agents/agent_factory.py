from typing import Dict, Type
from agents.conversation_agent import ConversationAgent
from agents.web_search_agent import WebSearchAgent  
from agents.base_agent import Agent


class AgentFactory:
    """
    A factory class responsible for creating agent instances based on the provided agent type.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.agent_registry: Dict[str, Type[Agent]] = {
            'conversation_agent': ConversationAgent,
            'web_search_agent': WebSearchAgent,  # Register the new agent
        }

    def factory(self, agent_type: str) -> Agent:
        """Factory method to create agents based on the agent_type string."""
        agent_class = self.agent_registry.get(agent_type)
        if agent_class is None:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return agent_class(**self.kwargs)
