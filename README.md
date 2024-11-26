# Agents

This repo is built for testing new agentic features. It is built from langchain and langgraph 

---

## Project Structure

```
agents/
├── __init__.py
├── agent_factory.py       # Factory for creating agent 
├── base_agent.py          # Abstract base class for all 
├── conversation_agent.py  # Agent for conversational 
├── llm_runnable.py        # Wrapper for LLM invocations
├── web_search_agent.py    # Agent that performs web 
├── tools/
│   ├── __init__.py
│   └── tool_registry.py   # Registry for managing tools
main.py                    # Entry point for running the application
requirements.txt           # Project dependencies
.env.template              # Template for environment variables
README.md                  # Project documentation
```

---

## Getting Started

### Prerequisites

- **Python 3.8 or higher**: Ensure you have Python installed. You can check your Python version with:

  ```bash
  python --version
  ```

- **Pip**: Python package installer should be available.

- **OpenAI API Key**: You need an API key from OpenAI. Sign up at [OpenAI](https://beta.openai.com/signup/).

- **Tavily API Key** (Optional): For web search capabilities, obtain a Tavily API key from [Tavily](https://tavily.com/).

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/llm-agents-project.git
   cd llm-agents-project
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### Setup Environment Variables

1. **Copy the Template `.env` File**

   ```bash
   cp .env.template .env
   ```

2. **Populate the `.env` File**

   Open the `.env` file and replace the placeholders with your actual API keys:

   ```dotenv
   OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
   TAVILY_API_KEY=YOUR_TAVILY_API_KEY_HERE
   ```

   - If you don't have a `TAVILY_API_KEY`, and don't plan to use the web search agent, you can leave it blank or remove it.

---

## Usage

### Running the Application

Run the `main.py` script to start interacting with the agents:

```bash
python main.py
```

### Selecting an Agent

Upon running the script, you'll be prompted to select an agent type:

```plaintext
Enter agent type (conversation_agent or web_search_agent):
```

- **conversation_agent**: An agent that engages in general conversation with memory support.
- **web_search_agent**: An agent that can perform web searches for up-to-date information.

Type the desired agent name and press Enter.

### Interacting with the Agent

After selecting an agent, you can start the conversation:

```plaintext
Start interacting with the agent (type 'exit' to stop):
You: Hello!
AI: Hello! How can I assist you today?
```

- **Exit the Conversation**: Type `exit` and press Enter to terminate the session.

---

## Extending the Project

### Adding New Agents

1. **Create a New Agent File**

   In the `agents` directory, create a new file for your agent, e.g., `my_custom_agent.py`.

2. **Implement the Agent Class**

   Your agent should inherit from `Agent` in `base_agent.py` and implement the `run` method.

   ```python
   # agents/my_custom_agent.py
   from .base_agent import Agent

   class MyCustomAgent(Agent):
       def __init__(self, OPENAI_API_KEY: str):
           # Initialize your agent here
           pass

       def run(self, input_text: str, chat_history: list = None):
           # Implement the agent's behavior here
           pass
   ```

3. **Register the Agent**

   Add your agent to the `AgentFactory` in `agent_factory.py`:

   ```python
   from agents.my_custom_agent import MyCustomAgent

   class AgentFactory:
       def __init__(self, **kwargs):
           self.kwargs = kwargs
           self.agent_registry = {
               'conversation_agent': ConversationAgent,
               'web_search_agent': WebSearchAgent,
               'my_custom_agent': MyCustomAgent,  # Register your new agent
           }
   ```

### Adding New Tools

1. **Create a New Tool File**

   In the `agents/tools` directory, create a new file for your tool, e.g., `my_custom_tool.py`.

2. **Implement the Tool Class**

   Your tool should inherit from LangChain's `BaseTool` or a similar base class.

   ```python
   # agents/tools/my_custom_tool.py
   from langchain.tools import BaseTool

   class MyCustomTool(BaseTool):
       def __init__(self, **kwargs):
           # Initialize your tool here
           pass

       def run(self, query: str):
           # Implement the tool's functionality here
           pass
   ```

3. **Register the Tool**

   Add your tool to the `ToolRegistry` in `tool_registry.py`:

   ```python
   # agents/tools/tool_registry.py
   from .my_custom_tool import MyCustomTool

   class ToolRegistry:
       def __init__(self):
           self.tool_registry = {
               'tavily_search': TavilySearchResults,
               'my_custom_tool': MyCustomTool,  # Register your new tool
           }
   ```

4. **Use the Tool in an Agent**

   Modify an agent to include your new tool:

   ```python
   # agents/my_custom_agent.py
   from .tools.tool_registry import ToolRegistry

   class MyCustomAgent(Agent):
       def __init__(self, OPENAI_API_KEY: str):
           tool_registry = ToolRegistry()
           self.tools = tool_registry.get_tools(['my_custom_tool'])
           # Continue with agent initialization
   ```

---
