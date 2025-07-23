# 导入 LangChain 的核心组件
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 配置模型 (关键部分) ---

# 1. 初始化 ChatOpenAI 模型实例
llm = ChatOpenAI(
    # model: 指定你想使用的模型名称
    # 你可以在硅基流动的模型广场找到所有支持的模型，例如国产的 Zhipu/chatglm3-6b 等
    model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",

    # api_key: 填入你在硅基流动平台获取的 API 密钥
    api_key="sk-dksdphixqcfngyjkcvzbyanwkzxltgmoccmxscmyyrnsrung",

    # base_url: 指定硅基流动的 OpenAI 兼容 API 端点
    base_url="https://api.siliconflow.cn/v1", # <--- 这是固定的地址

    # 其他参数，例如温度，控制模型输出的随机性
    temperature=0.7
)

# --- 构建和运行一个简单的链 (Chain) ---

# 2. 创建一个提示模板
#    这部分和调用 OpenAI 时完全一样
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识渊博的AI助手，擅长用简洁的语言解释复杂的技术概念。"),
    ("user", "{input}")
])

# 3. 创建一个简单的输出解析器，将结果转换为字符串
output_parser = StrOutputParser()

# 4. 使用 LangChain 表达式语言 (LCEL) 将组件连接成一个链
chain = prompt | llm | output_parser

# 5. 调用链并传入你的问题
print("正在调用硅基流动的模型...")
response = chain.invoke({"input": "请解释一下什么是“检索增强生成 (RAG)”？"})

# 6. 打印模型的返回结果
print("\n模型的回答：")
print(response)