import os
import dashscope
from openai import OpenAI
from langchain_core.embeddings import Embeddings

class QwenEmbeddings(Embeddings):
    def __init__(self, model="text-embedding-v4", batch_size=10):
        self.model = model
        self.batch_size = batch_size
        # 确保在使用类时已经加载了环境变量
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not found in environment variables.")
            
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    def embed_documents(self, texts):
        texts = [text.strip() for text in texts if text.strip()]
        if not texts: return []
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            embeddings.extend([item.embedding for item in response.data])
        return embeddings
    
    def embed_query(self, text):
        text = text.strip()
        if not text: return []
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding