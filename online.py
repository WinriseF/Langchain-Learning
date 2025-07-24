# 使AI具备上网能力
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    api_key="sk-dksdphixqcfngyjkcvzbyanwkzxltgmoccmxscmyyrnsrung",
    base_url="https://api.siliconflow.cn/v1"
)

# 定义工具箱，
tools = [DuckDuckGoSearchRun()]

# 3. 创建一个专门为 Agent 设计的 Prompt
# 它会告诉 LLM 它有哪些工具以及如何思考
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手。"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 创建 Agent
agent = create_tool_calling_agent(llm, tools, prompt)

# 创建 Agent 执行器 (Agent Executor)
# 这是真正运行 Agent 的引擎
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # verbose=True 会打印思考过程

# 运行 Agent
agent_executor.invoke({"input": "今天上海的天气怎么样？"})