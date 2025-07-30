import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader 
from langchain_community.tools import GoogleSerperRun
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.memory import ConversationBufferMemory

# 加载环境变量
load_dotenv()

llm_api_key = os.getenv("LLM")
serper_api_key = os.getenv("SERPER_API_KEY")

llm = ChatOpenAI(
    model="Qwen/Qwen3-8B", 
    api_key=llm_api_key,
    base_url="https://api.siliconflow.cn/v1",
    temperature=0
)

search_wrapper = GoogleSerperAPIWrapper()
search_tool = GoogleSerperRun(api_wrapper=search_wrapper)
search_tool.description = "当需要回答关于时事、最新信息或不确定事实的问题时，必须使用此工具进行网络搜索。"
python_repl_tool = PythonREPLTool()
@tool
def scrape_website(url: str) -> str:
    """当需要从一个特定的网页URL获取详细内容时，使用此工具。"""
    print(f"--- 正在浏览网页: {url} ---")
    loader = WebBaseLoader(url)
    docs = loader.load()
    return "".join(doc.page_content for doc in docs)
tools = [search_tool, python_repl_tool, scrape_website]


# --- 创建带记忆的 Agent 和执行器 ---
prompt = hub.pull("hwchase17/react-chat")
memory = ConversationBufferMemory(memory_key="chat_history")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    memory=memory,
    verbose=True,
    handle_parsing_errors=True 
)

print("\n--- 可以开始对话了 ---")
print("--- (输入 'exit', 'quit' 或 '再见' 来结束对话) ---")

while True:
    user_input = input("\n输入: ")

    if user_input.lower() in ["exit", "quit", "再见"]:
        print("再见！")
        break

    response = agent_executor.invoke({"input": user_input})
    print(f"\n回答: {response.get('output')}")