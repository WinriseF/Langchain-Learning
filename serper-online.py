import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_openai import ChatOpenAI
from langchain import hub

# 加载环境变量
load_dotenv()

llm = ChatOpenAI(
    model="THUDM/GLM-4-9B-0414",
    api_key=os.getenv("LLM"),
    base_url="https://api.siliconflow.cn/v1",
    temperature=0
)

tools = [GoogleSerperRun(api_wrapper=GoogleSerperAPIWrapper())]

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True 
)

question = "今天上海的天气怎么样？"
print(f"正在提问: {question}")

response = agent_executor.invoke({"input": question})
print("\n")
print(response.get("output"))
