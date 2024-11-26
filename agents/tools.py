from langchain_community.tools.tavily_search import TavilySearchResults

def initialize_tools():
    """
    Initialize tools for use with LangChain agents.
    Returns:
        List of tools.
    """
    # Tavily web search tool
    tavily_search_tool = TavilySearchResults(max_results=1)

    # Return a list of all tools
    return [tavily_search_tool]
