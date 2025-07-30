import os
import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    api_key="sk-izkjegktavpikceucoempxuvhxoqzfyhiaxyxchmsxubmjzm",
    base_url="https://api.siliconflow.cn/v1",
    temperature=0
)

# 定义数据结构
class Decision(BaseModel):
    """定义单个核心决定的数据结构"""
    item_name: str = Field(description="涉及的产品或项目名称，例如 '产品A' 或 'Project Phoenix'")
    details: str = Field(description="关于该项目的具体决定内容摘要")
    owner: Optional[str] = Field(description="该事项的负责人，如果没有明确指定则为空", default=None)
    due_date: Optional[str] = Field(description="相关的截止日期，例如 '8月15日' 或 '9月底前'", default=None)
    budget: Optional[float] = Field(description="相关的预算金额（仅抽取出数字部分），如果没有则为空", default=None)

class MeetingInfo(BaseModel):
    """定义整个会议纪要的顶层数据结构"""
    topic: str = Field(description="会议的核心主题")
    attendees: List[str] = Field(description="所有参会人员的姓名列表")
    date: datetime.date = Field(description="会议举行的日期，格式为 YYYY-MM-DD")
    decisions: List[Decision] = Field(description="会议中做出的所有核心决定的列表")


# ==============================================================================
# 创建一个知道如何解析上述“蓝图”的解析器
# ==============================================================================

parser = PydanticOutputParser(pydantic_object=MeetingInfo)

# ==============================================================================
# 构建一个包含“格式化指令”的提示模板
# ==============================================================================

prompt_template = """
你是一个专业的会议助理AI。请仔细阅读下面提供的会议纪要原文。
你的任务是根据原文内容，严格、准确地提取出所有关键信息。

{format_instructions}

会议纪要原文:
```{meeting_minutes}```

请直接输出符合上述格式的JSON对象，不要包含任何其他解释性文字。
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["meeting_minutes"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 创建链
chain = prompt | llm | parser

# 读取 meeting.txt 文件内容
with open('meeting.txt', 'r', encoding='utf-8') as f:
    meeting_content = f.read()
print("--- 已成功读取文件 ---")


# 执行链，并传入会议纪要内容
print("\n--- 正在调用 LLM 进行智能信息提取，请稍候... ---")
parsed_output = chain.invoke({"meeting_minutes": meeting_content})


# 输出提取结果
print("\n --- 信息提取成功！--- \n")

# `parsed_output` 现在是一个 MeetingInfo 对象
print("提取出的数据类型:", type(parsed_output))
print("-" * 20)

# 我们可以像操作任何 Python 对象一样访问它的属性
print(f"会议主题: {parsed_output.topic}")
print(f"参会人员: {parsed_output.attendees}")
print(f"会议日期: {parsed_output.date}")
print("-" * 20)

print("核心决定详情:")
for i, decision in enumerate(parsed_output.decisions):
    print(f"  决定 {i+1}:")
    print(f"    项目: {decision.item_name}")
    print(f"    内容: {decision.details}")
    print(f"    负责人: {decision.owner}")
    print(f"    截止日期: {decision.due_date}")
    print(f"    预算: {decision.budget}")
print("-" * 20)

print("\n--- 转换成标准 JSON 格式 ---")
json_output = parsed_output.model_dump_json(indent=2)
print(json_output)