from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

chat_llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    api_key="sk-dksdphixqcfngyjkcvzbyanwkzxltgmoccmxscmyyrnsrung",
    base_url="https://api.siliconflow.cn/v1",
    temperature=0
)

# 中文向量嵌入模型配置
embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key="sk-dksdphixqcfngyjkcvzbyanwkzxltgmoccmxscmyyrnsrung",
    base_url="https://api.siliconflow.cn/v1"
)

loader = PyPDFLoader("项目可行性分析报告.pdf")
docs = loader.load()

# 将文档分割成小块，以便进行有效的检索
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# --- 创建向量数据库 ---
print("正在创建向量数据库，请稍候...")
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
print("向量数据库创建完成！")

# --- 创建 RAG 链 (Chain) ---
retriever = vectorstore.as_retriever()

# 创建一个提示模板，指导模型如何利用检索到的上下文来回答问题
prompt = ChatPromptTemplate.from_template("""请你仅根据下面提供的上下文来回答问题。如果你在上下文中找不到答案，就说你不知道。

<context>
{context}
</context>

问题: {input}
""")

# 创建一个文档处理链，它负责将检索到的文档块填充到提示中
document_chain = create_stuff_documents_chain(chat_llm, prompt)

# 将检索器和文档链结合，创建完整的 RAG 链
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 提问
question = "项目使用的技术框架是什么？"
print(f"\n正在提问: {question}")

response = retrieval_chain.invoke({"input": question})

print("\n模型的回答:")
print(response["answer"])