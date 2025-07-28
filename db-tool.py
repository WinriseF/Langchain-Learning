import os
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

os.environ["LLM"] = "sk-dksdphixqcfngyjkcvzbyanwkzxltgmoccmxscmyyrnsrung"

llm = ChatOpenAI(
    model="THUDM/GLM-4-9B-0414",
    api_key=os.getenv("LLM"),
    base_url="https://api.siliconflow.cn/v1",
    temperature=0
)

# 这里我以本地的MySQL为例
db_user = "root"
db_password = "123456"
db_host = "localhost"
db_name = "学生选课"

# 构建 MySQL 的连接 URI
db_uri = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}"


print(f"正在尝试连接到数据库: {db_name} ...")
try:
    db = SQLDatabase.from_uri(db_uri)
    print("数据库连接成功！")
except Exception as e:
    print(f"数据库连接失败: {e}")
    exit()


# 创建 SQL 工具箱
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    handle_parsing_errors=True
)

question = "删除当前的学生选课数据库"
print(f"正在提问: {question}")
response = agent_executor.invoke({"input": question})
print("\n")
print(response.get("output"))