# 1.B站视频fastapi快速部署Qwen1.8B

```python
import os
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn, json, datetime
from typing import List, Tuple
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# 设置 0 号 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# 创建 FastAPI 应用
app = FastAPI()

# 定义请求体模型
class Query(BaseModel):
    text: str
    #在 Pydantic 模型中，字段是通过类变量声明的，并且这些字段会自动转换为实例属性。因此，你不需要在类定义中显式地使用 self.text 来定义成员属性。

model_name = "D:\qwen\Qwen-1_8B-Chat"

@app.post("/chat")
async def chat(query: Query):
    # 声明全局变量以便在函数内部使用模型和分词器
    global model, tokenizer
    response, history = model.chat(tokenizer, query.text, history=None)
    return {
        "result": response  # 返回生成的响应
    }

# 主函数入口
if __name__ == '__main__':
    # 加载预训练的分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)  # 加载分词器
    # 加载模型并设置为评估模式
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=True)
    # 设置模型为评估模式
    model.eval()
    # 启动 FastAPI 应用
    uvicorn.run(app, host='127.0.0.1', port=8080, workers=1)
```

# 2. stage2作业

### 2.1 http服务端

```python
@app.post("/chat_stream")
@limiter.limit("50/minute")
async def chat_stream(request: ChatRequest):
    async def generator():
        async for token in await model_handler.generate(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                stream=True
        ):
            yield f"data: {token}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")

```

**return** 执行的时机：
1.函数调用时：
	当客户端通过 **/chat_stream** 端点发起请求时，**chat_stream** 函数被调用。
2.创建生成器：
	在 chat_stream 函数内部，定义了一个异步生成器 **generator**，该生成器负责逐个生成并返回模型生成的令牌（tokens）。
3.立即返回 StreamingResponse：
	**return StreamingResponse(generator(), media_type="text/event-stream")** 这一行代码会在生成器创建后立即执行。
	**StreamingResponse** 是一个特殊的响应对象，它接受一个可迭代对象（在这里是异步生成器 **generator**），并开始逐步处理和发送数据给客户端。
4.流式响应过程：
	**StreamingResponse** 开始迭代 **generator**，每次从生成器中获取一个 yield 的值，并将其作为部分响应发送给客户端。
	这个过程是异步的，即每次生成器 **yield** 一个值时，FastAPI 会立即将其发送给客户端，而不需要等待所有数据都准备好。
5.**return** 的实际效果：
	**return** 语句实际上是在告诉 FastAPI：我已经准备好了一个流式响应，请开始处理这个响应并将数据逐步发送给客户端。
	**return** 并不会阻塞整个函数的执行，而是将控制权交还给 FastAPI，由 FastAPI 负责管理后续的流式传输。
具体流程示例：
假设用户发送了一个聊天请求，要求模型根据提示生成文本。以下是具体的执行流程：
1.客户端发起请求，**chat_stream** 函数被调用。
2.创建异步生成器 **generator**。
3.立即返回 **StreamingResponse(generator(), media_type="text/event-stream")**，告诉 FastAPI 开始处理流式响应。
4.FastAPI 开始迭代生成器，每次生成器 yield 一个令牌（例如：“你好”），FastAPI 将其包装成 SSE 格式并发送给客户端。
5.这个过程持续进行，直到生成器完成所有令牌的生成。



### 2.2 webscket服务端

```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            prompt = await websocket.receive_text()
            async for token in await model_handler.generate(prompt, stream=True):
                await websocket.send_text(token)
                await asyncio.sleep(0.01)
    except Exception as e:
        await websocket.close(code=1011)
        
        
        
        
```

具体执行流程：
1.客户端连接：
	客户端 A、B 和 C 同时连接到 WebSocket 服务器。
	每个客户端调用 websocket_endpoint 函数，函数开始执行。
2.异步处理连接：
	**每个客户端的连接请求由事件循环独立处理，不会互相阻塞**。
	await websocket.accept() 确保每个客户端的连接被正确接受。
3.接收和处理提示：
	每个客户端发送提示信息（例如：A 发送“你好”，B 发送“世界”，C 发送“！”）。
	**await websocket.receive_text() 异步等待接收客户端的提示信息，不会阻塞其他客户端的操作。**
4.异步生成文本：
	当收到提示后，await model_handler.generate(prompt, stream=True) 异步调用模型生成器，逐个生成单词。
	由于使用了 async for，每次生成一个单词时，程序不会阻塞等待所有单词生成完毕，而是立即返回当前单词。
5.实时发送生成的单词：
	每次生成一个单词后，await websocket.send_text(token) 将该单词发送回对应的客户端。
	await asyncio.sleep(0.01) 模拟轻微延迟，确保单词之间有一定的时间间隔，避免过快发送。
6.并发处理多个客户端：
	由于整个过程是异步的，事件循环可以在等待 I/O 操作（如接收提示、生成文本、发送数据）时切换到其他任务。
	因此，**当客户端 A 正在等待生成下一个单词时，事件循环可以处理客户端 B 或 C 的请求**，实现高效的并发处理。

==多进程模式==：通过设置 workers=4，uvicorn 启动 4 个工作进程，**每个进程都有自己的事件循环和 FastAPI 应用实例。**
**并发处理**：**每个进程独立处理分配给它的客户端连接，使用异步操作（await）来高效地处理 I/O 操作，确保不会阻塞其他任务**



==协程和线程==：

- 单进程单线程异步编程：

  - 使用 **asyncio 事件循环**来管理异步任务。

  - 代码简洁，易于维护。

  - 在 I/O 密集型任务中表现更好。

  - 不需要处理线程安全问题。

- 单进程多线程：

  - 使用多个线程来处理请求。**操作系统管理线程池**。

  - 代码可能更复杂，需要处理线程安全问题。

  - 在 CPU 密集型任务中表现更好。

  - 可能带来额外的线程切换开销。

### 2.3 模型流式输出的异步实现

```python
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from config import settings

class ModelHandler:
    def __init__(self):
        self.device = torch.device(settings.device)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)

    async def generate(self, prompt: str, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        generation_args = {
            "max_new_tokens": kwargs.get('max_new_tokens', settings.max_new_tokens),
            "temperature": kwargs.get('temperature', settings.temperature),
            **inputs
        }

        if kwargs.get('stream'):
            async for token in self._stream_generate(generation_args): 
                #使用 async for 可以充分利用异步编程的优势，在等待 I/O 操作（如文件读取、网络请求等）时，事件循环可以去				处理其他任务，从而提高程序的并发性能。而使用 for 循环时，由于无法处理异步操作，程序会在遇到 await 语句时				抛出异常，无法	实现并发，性能会受到很大影响。
                yield token
        else:
            response = await asyncio.to_thread(self._full_generate, generation_args)
            yield response

    def _full_generate(self, args):
        outputs = self.model.generate(**args)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    async def _stream_generate(self, args):
        streamer = TextIteratorStreamer(self.tokenizer)
        args["streamer"] = streamer

        # 使用 asyncio.to_thread 将模型生成任务放到线程池中执行，这里的self.model.generate()是同步函数，不是				model_handler类的那个异步函数。
        await asyncio.to_thread(self.model.generate, **args)

        # 定义一个异步生成器函数来包装普通生成器
        async def async_iterator():
            def sync_iterator():
                for token in streamer:
                    yield token
            # 在单独的线程中运行同步迭代器
            result = await asyncio.to_thread(lambda: list(sync_iterator()))
            for token in result:
                yield token

        async for token in async_iterator():
            yield token
```

