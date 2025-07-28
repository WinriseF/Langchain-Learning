import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader 
from langchain_community.tools import GoogleSerperRun
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_openai import ChatOpenAI
from langchain import hub

os.environ["LLM"] = "sk-dksdphixqcfngyjkcvzbyanwkzxltgmoccmxscmyyrnsrung"
os.environ["SERPER_API_KEY"] = "987e66fccb22e21b0caad2cb518fdc67287878ad" 

llm = ChatOpenAI(
    model="THUDM/GLM-4-9B-0414",
    api_key=os.getenv("LLM"),
    base_url="https://api.siliconflow.cn/v1",
    temperature=0
)
search_wrapper = GoogleSerperAPIWrapper()
search_tool = GoogleSerperRun(api_wrapper=search_wrapper)
search_tool.description = "当需要回答关于时事、最新信息或不确定事实的问题时，必须使用此工具进行网络搜索。"

python_repl_tool = PythonREPLTool()


@tool
def scrape_website(url: str) -> str:
    """
    当需要从一个特定的网页URL获取详细内容时，使用此工具。
    输入必须是一个有效的、完整的URL (例如 https://www.google.com)。
    """
    print(f"--- 正在浏览网页: {url} ---")
    loader = WebBaseLoader(url)
    docs = loader.load()
    return "".join(doc.page_content for doc in docs)

#    Agent 会根据问题和每个工具的描述来决定使用哪一个
tools = [search_tool, python_repl_tool, scrape_website]

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=False, 
    handle_parsing_errors=True 
)
question = "看看https://python.langchain.com/docs/integrations/tools/这个网页"
print(f"正在提问: {question}")
response = agent_executor.invoke({"input": question})
print("\n")
print(response.get("output"))
