# AI/ML Implementation Patterns

## LLM Integration Patterns

### 1. Simple Chat Completion

```python
# Production-ready chat with error handling
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

class ChatService:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def chat(
        self,
        messages: list[dict],
        model: str = "gpt-4",
        temperature: float = 0.7
    ) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            raise
```

### 2. Streaming Responses

```python
# Backend streaming with SSE
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        try:
            async for chunk in openai_client.chat.completions.create(
                model="gpt-4",
                messages=request.messages,
                stream=True
            ):
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'content': content})}\n\n"
            
            yield "data: {\"done\": true}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

```typescript
// Frontend streaming consumer
const streamChat = async (message: string) => {
    const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: [{ role: 'user', content: message }] })
    })

    const reader = response.body?.getReader()
    const decoder = new TextDecoder()

    while (true) {
        const { done, value } = await reader!.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6))
                if (data.content) {
                    updateChatContent(data.content)
                }
                if (data.done) return
            }
        }
    }
}
```

### 3. Function Calling / Tool Use

```python
# Function calling with structured outputs
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
]

async def chat_with_tools(message: str):
    messages = [{"role": "user", "content": message}]
    
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    message = response.choices[0].message
    
    if message.tool_calls:
        # Execute function
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Execute the actual function
            result = await execute_function(function_name, function_args)
            
            # Add function result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
        
        # Get final response with function results
        final_response = await client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        return final_response.choices[0].message.content
    
    return message.content
```

## RAG (Retrieval-Augmented Generation) Patterns

### 1. Basic RAG Pipeline

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAGPipeline:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Pinecone.from_existing_index(
            "document-index",
            self.embeddings
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    async def index_documents(self, documents: list[str]):
        """Index documents into vector store"""
        chunks = self.text_splitter.split_documents(documents)
        await self.vector_store.aadd_documents(chunks)
    
    async def query(self, question: str, k: int = 5) -> str:
        """Query with retrieval"""
        # Retrieve relevant documents
        docs = await self.vector_store.asimilarity_search(
            question,
            k=k
        )
        
        # Build context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate response
        messages = [
            {
                "role": "system",
                "content": "Answer based on the context provided."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
        
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        return response.choices[0].message.content
```

### 2. Advanced RAG with Reranking

```python
from cohere import Client as CohereClient

class AdvancedRAG:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Pinecone.from_existing_index("docs", self.embeddings)
        self.reranker = CohereClient(api_key=os.getenv("COHERE_API_KEY"))
        self.cache = Redis()
    
    async def query(self, question: str) -> dict:
        # Check cache
        cache_key = f"rag:{hashlib.md5(question.encode()).hexdigest()}"
        cached = await self.cache.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Retrieve more documents than needed
        docs = await self.vector_store.asimilarity_search(question, k=20)
        
        # Rerank for better relevance
        reranked = self.reranker.rerank(
            query=question,
            documents=[doc.page_content for doc in docs],
            top_n=5,
            model="rerank-english-v2.0"
        )
        
        # Get top reranked documents
        top_docs = [docs[result.index] for result in reranked.results]
        context = "\n\n".join([doc.page_content for doc in top_docs])
        
        # Generate with citations
        response = await self.generate_with_citations(question, top_docs)
        
        # Cache result
        await self.cache.setex(cache_key, 3600, json.dumps(response))
        
        return response
    
    async def generate_with_citations(self, question: str, docs: list) -> dict:
        # Number the sources
        context = "\n\n".join([
            f"[{i+1}] {doc.page_content}"
            for i, doc in enumerate(docs)
        ])
        
        messages = [
            {
                "role": "system",
                "content": "Answer using the provided sources. Cite sources using [1], [2], etc."
            },
            {
                "role": "user",
                "content": f"Sources:\n{context}\n\nQuestion: {question}"
            }
        ]
        
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        return {
            "answer": response.choices[0].message.content,
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
        }
```

### 3. Hybrid Search (Semantic + Keyword)

```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever

class HybridRAG:
    def __init__(self, documents: list):
        # Vector retriever (semantic)
        self.vector_retriever = Pinecone.from_documents(
            documents,
            OpenAIEmbeddings()
        ).as_retriever(search_kwargs={"k": 10})
        
        # BM25 retriever (keyword)
        self.keyword_retriever = BM25Retriever.from_documents(documents)
        self.keyword_retriever.k = 10
        
        # Ensemble retriever (combines both)
        self.retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.keyword_retriever],
            weights=[0.6, 0.4]  # Favor semantic search slightly
        )
    
    async def query(self, question: str) -> str:
        docs = await self.retriever.aget_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate response
        messages = [
            {"role": "system", "content": "Answer based on context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQ: {question}"}
        ]
        
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        return response.choices[0].message.content
```

## Model Fine-Tuning Patterns

### 1. LoRA Fine-Tuning

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

def setup_lora_model(base_model: str):
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,  # Quantization for efficiency
        device_map="auto"
    )
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # LoRA rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def train_lora(model, train_dataset, output_dir: str):
    from transformers import TrainingArguments, Trainer
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        save_steps=500,
        logging_steps=100,
        warmup_steps=100
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
```

## Model Deployment Patterns

### 1. FastAPI Model Serving

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

class PredictionRequest(BaseModel):
    text: str
    max_length: int = 100

class PredictionResponse(BaseModel):
    generated_text: str
    tokens_used: int

app = FastAPI()

# Load model once at startup
@app.on_event("startup")
async def load_model():
    global model, tokenizer
    model = AutoModelForCausalLM.from_pretrained("model-path")
    tokenizer = AutoTokenizer.from_pretrained("model-path")
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Tokenize
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return PredictionResponse(
            generated_text=generated_text,
            tokens_used=len(outputs[0])
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}
```

### 2. Batch Processing Pipeline

```python
import asyncio
from typing import List

class BatchProcessor:
    def __init__(self, model, batch_size: int = 32):
        self.model = model
        self.batch_size = batch_size
        self.queue = asyncio.Queue()
    
    async def process_batch(self, items: List[str]) -> List[str]:
        """Process a batch of items"""
        inputs = tokenizer(
            items,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=100)
        
        return [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
    
    async def worker(self):
        """Background worker that processes batches"""
        batch = []
        
        while True:
            try:
                # Wait for item or timeout
                item = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=1.0
                )
                batch.append(item)
                
                # Process when batch is full or queue is empty
                if len(batch) >= self.batch_size or self.queue.empty():
                    if batch:
                        texts = [item["text"] for item in batch]
                        results = await self.process_batch(texts)
                        
                        # Set results
                        for item, result in zip(batch, results):
                            item["future"].set_result(result)
                        
                        batch = []
            
            except asyncio.TimeoutError:
                # Process remaining items
                if batch:
                    texts = [item["text"] for item in batch]
                    results = await self.process_batch(texts)
                    for item, result in zip(batch, results):
                        item["future"].set_result(result)
                    batch = []
    
    async def predict(self, text: str) -> str:
        """Add item to queue and wait for result"""
        future = asyncio.Future()
        await self.queue.put({"text": text, "future": future})
        return await future
```

## Monitoring & Observability

```python
from prometheus_client import Counter, Histogram
import time

# Metrics
llm_requests = Counter('llm_requests_total', 'Total LLM requests')
llm_errors = Counter('llm_errors_total', 'Total LLM errors')
llm_latency = Histogram('llm_latency_seconds', 'LLM request latency')
llm_tokens = Counter('llm_tokens_total', 'Total tokens used')

async def monitored_chat(messages: list) -> str:
    llm_requests.inc()
    start_time = time.time()
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        # Record metrics
        duration = time.time() - start_time
        llm_latency.observe(duration)
        llm_tokens.inc(response.usage.total_tokens)
        
        return response.choices[0].message.content
    
    except Exception as e:
        llm_errors.inc()
        logger.error(f"LLM error: {e}")
        raise
```
