import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
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

tools = [DuckDuckGoSearchRun()]

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True) #verbose=True显示过程

question = "最近的中国马上将要建造的世界最大的水电站是什么？"
print(f"正在提问: {question}")

print(agent_executor.invoke({"input": question})["output"])