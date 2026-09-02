import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# 加载环境变量（把你的 API Key 放在 .env 文件里）
load_dotenv()
print("正在加载模型！")
# 初始化大模型（这里以通义千问为例，你也可以换成 gpt-4o、claude-3.5-sonnet 等）
llm = init_chat_model(model="deepseek-v4-flash")
print("模型成功加载！")

if __name__ == "__main__":
    response = llm.invoke("你好请问你是？")
    print(response.content)