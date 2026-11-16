# Import packages
from langchain.messages import AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage
import os
from langchain.tools import tool
import time
from dotenv import load_dotenv


load_dotenv()

# Connect notebook to langsmith
os.environ['LANGSMITH_TRACING'] = os.getenv('LANGSMITH_TRACING')
os.environ['LANGSMITH_ENDPOINT'] = os.getenv('LANGSMITH_ENDPOINT')
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT')
os.environ['DEEPSEEK_API_KEY'] = os.getenv('DEEPSEEK_API_KEY')



if __name__ == "__main__":

    # Create a subagent specialising in python
    subagent_python = create_agent(model="deepseek:deepseek-v4-pro")

    # Wrap it as a tool
    @tool("python", description="Python developer")
    def call_python_agent(query: str):
        result = subagent_python.invoke({"messages": [{"role": "user", "content": query}]})
        return result["messages"][-1].content

    # Create a subagent specialising in networking
    subagent_network = create_agent(model="deepseek:deepseek-v4-pro")

    # Wrap it as a tool
    @tool("networking", description="Network expert")
    def call_network_agent(query: str):
        result = subagent_network.invoke({"messages": [{"role": "user", "content": query}]})
        return result["messages"][-1].content

    # Create a subagent specialising in Teminal GUI
    subagent_gui = create_agent(model="deepseek:deepseek-v4-pro")

    # Wrap it as a tool
    @tool("gui", description="Develop a Terminal UI using Python")
    def call_gui_agent(query: str):
        result = subagent_gui.invoke({"messages": [{"role": "user", "content": query}]})
        return result["messages"][-1].content


    # Create a subagent specialising in code review
    subagent_reviewer = create_agent(model="deepseek:deepseek-v4-pro")

    # Wrap it as a tool
    @tool("review", description="Expert code reviewer")
    def call_review_agent(query: str):
        result = subagent_gui.invoke({"messages": [{"role": "user", "content": query}]})
        return result["messages"][-1].content

    # Main agent with subagent as a tool
    main_agent = create_agent(model="deepseek:deepseek-v4-flash", tools=[call_python_agent, call_network_agent, call_gui_agent, call_review_agent])

    print("Main Agent is ready.", flush=True)
    time.sleep(3)
    print("Please enter prompt", flush=True)
    prompt = str(input()).lower()

    result = main_agent.invoke({"messages": [{"role": "user", "content": f"{prompt}"}]})
    print(result["structured_response"])