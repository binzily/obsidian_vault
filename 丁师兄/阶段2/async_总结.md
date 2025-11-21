###  1.Python高级编程

#### 1.1 yield

yield 是生成器的核心关键字，用于定义生成器函数。带yield的函数类似于**被装饰器修饰**的函数，它有两个主要功能：

- **暂停和恢复函数执行**：当函数执行到 yield 时，函数会暂停执行，并将 yield 后面的值返回给调用者。当再次调用生成器的 .send() 或 next() 方法时，函数会从上次暂停的地方继续执行。
- **双向通信**：yield 不仅可以返回值，还可以接收从外部传入的值（通过 .send() 方法）。

```python
def generator_example(): # 生成器函数
    print("开始执行")
    x = yield 1  # 返回1，并暂停
    print("收到值：", x)
    yield x + 1  # 返回x+1，并暂停

gen = generator_example() # 并不会执行函数，这句代码是创建一个生成器对象。
print(next(gen))  # 输出：开始执行，返回1
print(gen.send(10))  # 输出：收到值：10，返回11
```

**生成器对象的创建**：

```python
gen = generator_example()
```

当执行这行代码时，它只是创建了一个生成器对象 `gen`，此时生成器函数 `generator_example` 内部的代码并没有开始执行。

**生成器的首次启动**：

```python
print(next(gen))
```

调用 `next(gen)` 会启动生成器，开始执行 `generator_example` 函数体内的代码。函数先打印 `"开始执行"`，接着遇到 `x = yield 1` 这行代码。这里，`yield` 关键字起到了两个作用：

- 它会返回值 `1` 给调用者（也就是 `next(gen)` 的调用处），所以 `print(next(gen))` 会输出 `1`。
- 同时，生成器函数会暂停执行，等待下一次的调用。此时，变量 `x` 还没有被赋值。

**通过 `send` 方法传递值并继续执行生成器**：

```python
print(gen.send(10))
```

调用 `gen.send(10)` 时，它会做两件事：

- 把值 `10` 传递给生成器函数中最近一次暂停的 `yield` 表达式处，也就是将 `10` 赋值给变量 `x`。
- 继续执行生成器函数中剩余的代码。所以，接下来会打印 `"收到值： 10"`，然后执行 `yield x + 1`，也就是 `yield 10 + 1`，返回 `11` 给调用者，因此 `print(gen.send(10))` 会输出 `11`。

#### 1.2 yield from

**不用 `yield from` 的情况**

先看一个不使用 `yield from` 来嵌套生成器的例子：

```python
def sub_generator():
    yield 1
    yield 2

def main_generator():
    for value in sub_generator():
        yield value

gen = main_generator()
print(next(gen))  # 输出: 1
print(next(gen))  # 输出: 2
```

**代码解释**

- `sub_generator` 是一个简单的生成器函数，它可以生成两个值 `1` 和 `2`。
- `main_generator` 函数里使用 `for` 循环遍历 `sub_generator` 生成器对象，将 `sub_generator` 生成的每个值通过 `yield` 语句再传递出来。

**执行流程**

1. 当调用 `main_generator()` 时，返回一个 `main_generator` 的生成器对象 `gen`。
2. 第一次调用 `next(gen)`，进入 `main_generator` 函数，开始 `for` 循环，**`for` 循环会调用 `sub_generator` 的 `next()` 方法**获取第一个值 `1`，然后通过 `yield` 把 `1` 传递出来，所以打印出 `1`。
3. 第二次调用 `next(gen)`，`for` 循环继续调用 `sub_generator` 的 `next()` 方法获取第二个值 `2`，再通过 `yield` 传递出来，打印出 `2`。

**使用 `yield from` 的情况**

再看使用 `yield from` 的代码：

```python
def sub_generator():
    yield 1
    yield 2

def main_generator():
    yield from sub_generator()
    #   for value in sub_generator(): for循环相当于next(sub_generator())
    #   	yield value

gen = main_generator()
print(next(gen))  # 输出: 1
print(next(gen))  # 输出: 2
```

**代码解释**

- `sub_generator` 依然是那个可以生成 `1` 和 `2` 的生成器函数。
- `main_generator` 函数中使用了 `yield from sub_generator()`，这行代码的作用就是把 `sub_generator` 生成器对象委托给 `main_generator`。

**执行流程**

1. 调用 `main_generator()` 得到 `main_generator` 的生成器对象 `gen`。
2. 第一次调用 `next(gen)`，`main_generator` 直接把控制权交给 `sub_generator`，`sub_generator` 开始执行，遇到第一个 `yield 1`，返回 `1`，就好像这个 `1` 是 `main_generator` 生成的一样，所以打印出 `1`。
3. 第二次调用 `next(gen)`，`sub_generator` 继续执行，遇到第二个 `yield 2`，返回 `2`，同样看起来就像是 `main_generator` 生成的，打印出 `2`。

**`yield from` 的优势**

- **代码简洁**：使用 `yield from` 避免了写 `for` 循环来遍历子生成器，代码更加简洁易读。
- **处理异常和返回值更方便**：`yield from` 不仅能传递子生成器的值，还能处理子生成器抛出的异常和返回值，就像之前例子里 `sub_generator` 返回 `3` 能被 `main_generator` 接收一样。

综上所述，`yield from sub_generator()` 让 `main_generator` 可以直接把 `sub_generator` 生成的值传递出来，就如同这些值是 `main_generator` 自身生成的，并且还提供了更便捷的异常和返回值处理能力。

​	**生成器执行到return语句时就会抛出stopiteration异常,并且return的值会被保存为异常对象的value属性。**由于生成器是为了逐个生成值而设计的，当它遇到 `return` 语句时，意味着它已经完成了生成值的任务，没有更多的值可以生成了。

#### 1.3 协程

**corotine的本质是一个简单的生成器，它可以yield一个东西，可以return一个value，return相当于raise stopitration。**

​	yield如何结合@asyncio.coroutine实现协程?在协程中，**只要是和IO任务类似的、耗费时间的任务都需要使用`yield from`来进行中断，达到异步功能！**

```python
# 使用同步方式编写异步功能
import time
import asyncio
@asyncio.coroutine # 标志协程的装饰器
def taskIO_1():
    print('开始运行IO任务1...')
    yield from asyncio.sleep(2)  # 假设该任务耗时2s
    print('IO任务1已完成，耗时2s')
    return taskIO_1.__name__
@asyncio.coroutine # 标志协程的装饰器
def taskIO_2():
    print('开始运行IO任务2...')
    yield from asyncio.sleep(3)  # 假设该任务耗时3s
    print('IO任务2已完成，耗时3s')
    return taskIO_2.__name__
@asyncio.coroutine # 标志协程的装饰器
def main(): # 调用方
    tasks = [taskIO_1(), taskIO_2()]  # 把所有任务添加到task中
    done,pending = yield from asyncio.wait(tasks) # 子生成器
    for r in done: # done和pending都是一个任务，所以返回结果需要逐个调用result()
        print('协程无序返回值：'+r.result())

if __name__ == '__main__':
    start = time.time()
    loop = asyncio.get_event_loop() # 创建一个事件循环对象loop
    try:
        loop.run_until_complete(main()) # 完成事件循环，直到最后一个任务结束
    finally:
        loop.close() # 结束事件循环
    print('所有IO任务总耗时%.5f秒' % float(time.time()-start))
```

执行结果如下：

```python
开始运行IO任务1...
开始运行IO任务2...
IO任务1已完成，耗时2s
IO任务2已完成，耗时3s
协程无序返回值：taskIO_2
协程无序返回值：taskIO_1
所有IO任务总耗时3.00209秒
```

1）上面代码先通过`get_event_loop()`**获取**了一个**标准事件循环**loop(因为是一个，所以协程是单线程)

2）然后，我们通过`run_until_complete(main())`来运行协程(此处把调用方协程main()作为参数，调用方负责调用其他委托生成器)，`run_until_complete`的特点就像该函数的名字，直到循环事件的所有事件都处理完才能完整结束。

3）进入调用方协程，我们把多个任务[`taskIO_1()`和`taskIO_2()`]放到一个`task`列表中，可理解为打包任务。

4）现在，我们使用`asyncio.wait(tasks)`来获取一个**awaitable objects即可等待对象的集合**(此处的aws是协程的列表)，**并发运行传入的aws**，同时通过`yield from`返回一个包含`(done, pending)`的元组，**done表示已完成的任务列表，pending表示未完成的任务列表**；如果使用`asyncio.as_completed(tasks)`则会按完成顺序生成协程的**迭代器**(常用于for循环中)，因此当你用它迭代时，会尽快得到每个可用的结果。【此外，当**轮询**到某个事件时(如taskIO_1())，直到**遇到**该**任务中的`yield from`中断**，开始**处理下一个事件**(如taskIO_2()))，当`yield from`后面的子生成器**完成任务**时，该事件才再次**被唤醒**】

5）因为`done`里面有我们需要的返回结果，但它目前还是个任务列表，所以要取出返回的结果值，我们遍历它并逐个调用`result()`取出结果即可。(注：对于`asyncio.wait()`和`asyncio.as_completed()`返回的结果均是先完成的任务结果排在前面，所以此时打印出的结果不一定和原始顺序相同，但使用`gather()`的话可以得到原始顺序的结果集）

6）最后我们通过`loop.close()`关闭事件循环。

综上所述：协程的完整实现是靠**①事件循环＋②协程**。



- **@asyncio.coroutine = async**
- **yield from = await**
- **当await一个coroutine时，发生了：向生成器一样调用corotine，运行coro并返回**。因为await = yield from，这是在没有event loop的情况。有event loop时，遇到io操作（asyncio.sleep()）是不会等的，是看谁先结束那就继续执行谁。
- **await + 可等待对象（协程对象，task对象，future对象）**

代码简化为：

```python
import time
import asyncio
async def taskIO_1():
    print('开始运行IO任务1...')
    await asyncio.sleep(2)  # 假设该任务耗时2s
    print('IO任务1已完成，耗时2s')
    return taskIO_1.__name__
async def taskIO_2():
    print('开始运行IO任务2...')
    await asyncio.sleep(3)  # 假设该任务耗时3s
    print('IO任务2已完成，耗时3s')
    return taskIO_2.__name__
async def main(): # 调用方
    tasks = [taskIO_1(), taskIO_2()]  # 把所有任务添加到task中
    done,pending = await asyncio.wait(tasks) # 子生成器，不能直接await(tasks)
    for r in done: # done和pending都是一个任务，所以返回结果需要逐个调用result()
        print('协程无序返回值：'+r.result())

if __name__ == '__main__':
    start = time.time()
    loop = asyncio.get_event_loop() # 创建一个事件循环对象loop
    try:
        loop.run_until_complete(main()) # 完成事件循环，直到最后一个任务结束
    finally:
        loop.close() # 结束事件循环
    print('所有IO任务总耗时%.5f秒' % float(time.time()-start))
```

由于 `await` 只能等待单个可等待对象，而不能直接处理任务列表，所以不能直接await（tasks），需要使用 `asyncio.wait()` 或 `asyncio.gather()` 等函数来并发执行多个任务。`asyncio.wait()` 可以让你更好地控制和处理任务的完成状态，区分已完成和未完成的任务。





#### 1.4 asyncio的不同方法实现协程

##### 1.4.1 asyncio.wait()

你可以将一个操作分成多个部分并分开执行，而`wait(tasks)`可以被用于**中断**任务集合(tasks)中的某个**被事件循环轮询**到的**任务**，直到该协程的其他后台操作完成才**被唤醒**。

```python
import time
import asyncio
async def taskIO_1():
    print('开始运行IO任务1...')
    await asyncio.sleep(2)  # 假设该任务耗时2s
    print('IO任务1已完成，耗时2s')
    return taskIO_1.__name__
async def taskIO_2():
    print('开始运行IO任务2...')
    await asyncio.sleep(3)  # 假设该任务耗时3s
    print('IO任务2已完成，耗时3s')
    return taskIO_2.__name__
async def main(): # 调用方
    tasks = [taskIO_1(), taskIO_2()]  # 把所有任务添加到task中
    done,pending = await asyncio.wait(tasks) # 子生成器
    for r in done: # done和pending都是一个任务，所以返回结果需要逐个调用result()
        print('协程无序返回值：'+r.result())

if __name__ == '__main__':
    start = time.time()
    loop = asyncio.get_event_loop() # 创建一个事件循环对象loop
    try:
        loop.run_until_complete(main()) # 完成事件循环，直到最后一个任务结束
    finally:
        loop.close() # 结束事件循环
    print('所有IO任务总耗时%.5f秒' % float(time.time()-start))
```

执行结果如下：

```python
开始运行IO任务1...
开始运行IO任务2...
IO任务1已完成，耗时2s
IO任务2已完成，耗时3s
协程无序返回值：taskIO_2
协程无序返回值：taskIO_1
所有IO任务总耗时3.00209秒
```

此处并发运行传入的`aws`(awaitable objects)，同时通过`await`返回一个包含(done, pending)的元组，**done**表示**已完成**的任务列表，**pending**表示**未完成**的任务列表。

**注：**

①只有当给`wait()`传入`timeout`参数时才有可能产生`pending`列表。

②通过`wait()`返回的**结果集**是**按照**事件循环中的任务**完成顺序**排列的，所以其往往**和原始任务顺序不同**。

##### 1.4.2 asyncio.gather()

如果你只关心协程并发运行后的结果集合，可以使用`gather()`，它不仅通过`await`返回仅一个结果集，而且这个结果集的**结果顺序**是传入任务的**原始顺序**。

```python
import time
import asyncio
async def taskIO_1():
    print('开始运行IO任务1...')
    await asyncio.sleep(3)  # 假设该任务耗时3s
    print('IO任务1已完成，耗时3s')
    return taskIO_1.__name__
async def taskIO_2():
    print('开始运行IO任务2...')
    await asyncio.sleep(2)  # 假设该任务耗时2s
    print('IO任务2已完成，耗时2s')
    return taskIO_2.__name__
async def main(): # 调用方
    resualts = await asyncio.gather(taskIO_1(), taskIO_2()) # 子生成器
    print(resualts)

if __name__ == '__main__':
    start = time.time()
    loop = asyncio.get_event_loop() # 创建一个事件循环对象loop
    try:
        loop.run_until_complete(main()) # 完成事件循环，直到最后一个任务结束
    finally:
        loop.close() # 结束事件循环
    print('所有IO任务总耗时%.5f秒' % float(time.time()-start))
```

执行结果如下：

```
开始运行IO任务2...
开始运行IO任务1...
IO任务2已完成，耗时2s
IO任务1已完成，耗时3s
['taskIO_1', 'taskIO_2']
所有IO任务总耗时3.00184秒
```

【解释】：`gather()`通过`await`直接**返回**一个结果集**列表**，我们可以清晰的从执行结果看出来，虽然任务2是先完成的，但最后返回的**结果集的顺序是按照初始传入的任务顺序排的**。

##### 1.4.3 asyncio.as_completed()

`as_completed(tasks)`是一个生成器，它管理着一个**协程列表**(此处是传入的tasks)的运行。当任务集合中的某个任务率先执行完毕时，会率先通过`await`关键字返回该任务结果。可见其返回结果的顺序和`wait()`一样，均是按照**完成任务顺序**排列的。

```python
import time
import asyncio
async def taskIO_1():
    print('开始运行IO任务1...')
    await asyncio.sleep(3)  # 假设该任务耗时3s
    print('IO任务1已完成，耗时3s')
    return taskIO_1.__name__
async def taskIO_2():
    print('开始运行IO任务2...')
    await asyncio.sleep(2)  # 假设该任务耗时2s
    print('IO任务2已完成，耗时2s')
    return taskIO_2.__name__
async def main(): # 调用方
    tasks = [taskIO_1(), taskIO_2()]  # 把所有任务添加到task中
    for completed_task in asyncio.as_completed(tasks):
        resualt = await completed_task # 子生成器
        print('协程无序返回值：'+resualt)

if __name__ == '__main__':
    start = time.time()
    loop = asyncio.get_event_loop() # 创建一个事件循环对象loop
    try:
        loop.run_until_complete(main()) # 完成事件循环，直到最后一个任务结束
    finally:
        loop.close() # 结束事件循环
    print('所有IO任务总耗时%.5f秒' % float(time.time()-start))
```

执行结果如下：

```
开始运行IO任务2...
开始运行IO任务1...
IO任务2已完成，耗时2s
协程无序返回值：taskIO_2
IO任务1已完成，耗时3s
协程无序返回值：taskIO_1
所有IO任务总耗时3.00300秒
```

【解释】：从上面的程序可以看出，使用`as_completed(tasks)`和`wait(tasks)`**相同之处**是返回结果的顺序是**协程的完成顺序**，这与gather()恰好相反。而**不同之处**是`as_completed(tasks)`可以**实时返回**当前完成的结果，而`wait(tasks)`需要等待所有协程结束后返回的`done`去获得结果。

##### 1.4.4 总结

以下`aws`指：`awaitable objects`。即**可等待对象集合**，如一个协程是一个可等待对象，一个装有多个协程的**列表**是一个`aws`。

| asyncio        | 主要传参 | 返回值顺序   | `await`返回值类型                   | 函数返回值类型 |
| -------------- | -------- | ------------ | ----------------------------------- | -------------- |
| wait()         | aws      | 协程完成顺序 | (done,pending) 装有两个任务列表元组 | coroutine      |
| as_completed() | aws      | 协程完成顺序 | 原始返回值                          | 迭代器         |
| gather()       | *aws     | 传参任务顺序 | 返回值列表                          | awaitable      |



#### 1.5 装饰器

```python
# 1定义一个装饰器(装饰器的本质是闭包)
def check(fn):
    def inner():
        print("请先登陆")
        fn()
    return inner

# 2使用装饰器装饰函数（增加一个登陆功能）
# 解释器遇到@check 会立即执行 comment = check(comment)
@check
def comment():
    print("发表评论")
```

```python
# 定义类装饰器
class Check(object):
    def __init__(self, fn):  # fn = comment
        self.__fn = fn

    def __call__(self, *args, **kwargs):
        print("登陆")
        self.__fn()

# 被装饰的函数
@Check  # comment = Check(comment)
def comment():
    print("发表评论")

comment()
```

### 2.线程、进程

并发和并行

- **并发**：在一段时间内**交替**执行多个任务。（单核cpu一般并发）

- **并行**：在一段时间内**真正的同时**一起执行多个任务。（任务数量小于cpu核心数）

  在同步中没有并发或者并行的概念。 

Python并发编程有三种方式 ：

**多线程（Thread）**，**多进程（Process）**，**多协程（Coroutine）**

#### 2.1二者的定义

##### 2.1.1 进程

​	一个正在运行的程序就是一个进程。

![image-20250209220330842](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20250209220330842.png)

​	进程的创建步骤：

```python
#导入进程模块
import multiprocessing
import time

def coding():
    for i in range(3):
        print("coding...")
        time.sleep(0.2)

def music():
    for i in range(3):
        print("music...")
        time.sleep(0.2)

if __name__ == '__main__':
    #通过进程类创建进程对象
    coding_process = multiprocessing.Process(target=coding)
    music_process = multiprocessing.Process(target=music)
    #启动进程
    coding_process.start()
    music_process.start()
```

```python
#代码运行结果
music...
coding...
music...
coding...
music...
coding...
#两个函数是一起运行的
```

​	进程间是不共享全局变量的。实际上创建一个子进程就是**把主进程的资源（例如全局变量）进行拷贝产生了一个新的进程**,这里主进程和子进程是互相独立的。

##### 2.1.2 线程

​	进程是分配资源的最小单位，一旦创建一个进程就会分配一定的资源，就像跟两个人聊 QQ 就需要打开两个 QQ 软件一样是比较浪费资源的。

​	线程是**程序执行的最小单位**，实际上进程只负责分配资源，而利用这些资源执行程序的是线程，也就说进程是线程的容器，一个进程中最少有一个线程来负责执行程序。同时线程自己不拥有系统资源，只需要一点儿在运行中必不可少的资源，但它可与同属一个进程的其它线程共享进程所拥有的全部资源。这就像通过一个 QQ 软件 (一个进程) 打开两个窗口 (两个线程) 跟两个人聊天一样，实现多任务的同时也节省了资源。

![image-20250210164144740](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20250210164144740.png)

这个图中，如果函数a和b都是操作的同一个全局变量呢？会造成线程间资源竞争问题。 可以用互斥锁解决。

​	进程的创建步骤：

```python
import time
import threading

# 编写代码
def coding():
    for i in range(3):
        print("coding...")
        time.sleep(0.2)

# 听音乐
def music():
    for i in range(3):
        print("music...")
        time.sleep(0.2)

if __name__ == '__main__':
    # 创建子线程
    coding_thread = threading.Thread(target=coding)
    music_thread = threading.Thread(target=music)

    # 启动子线程执行任务
    coding_thread.start()
    music_thread.start()
```

如果不设置参数，主线程/主进程会等待子线程/子进程结束后在结束。

线程之间执行是无序的，由cpu调度决定。

##### 2.1.3 进程线程的对比

关系： 1.线程是依附在进程里面的，没有进程就没有线程。 

​	2.一个进程默认提供一条线程，进程可以创建多个线程。

区别：1.进程之间不共享全局变量 

2. 线程之间共享全局变量，但是要注意资源竞争的问题，解决办法: 互斥锁或者线程同步 
3. 创建进程的资源开销要比创建线程的资源开销要大 
4.  进程是操作系统资源分配的基本单位，线程是CPU调度的基本单位 
5. 线程不能够独立执行，必须依存在进程中

优缺点：一个能使用多核一个不能。

#### 2.2三者的区别

最后我们将整个过程串一遍。 【引出问题】：

1. 同步编程的并发性不高
2. **多进程**编程效率受CPU核数限制，当任务数量远大于CPU核数时，执行效率会降低。
3. **多线程**编程需要线程之间的通信，而且需要**锁机制**来防止**共享变量**被不同线程乱改，而且由于Python中的**GIL(全局解释器锁)**，所以实际上也无法做到真正的并行。

【产生需求】：

1. 可不可以采用**同步**的方式来**编写异步**功能代码？
2. 能不能只用一个**单线程**就能做到不同任务间的切换？这样就没有了线程切换的时间消耗，也不用使用锁机制来削弱多任务并发效率！
3. 对于IO密集型任务，可否有更高的处理方式来节省CPU等待时间？

【结果】：所以**协程**应运而生。当然，实现协程还有其他方式和函数，以上仅展示了一种较为常见的实现方式。此外，**多进程和多线程是内核级别**的程序，而**协程是函数级别**的程序，是可以通过程序员进行调用的。以下是协程特性的总结：

| 协程           | 属性                                                     |
| -------------- | -------------------------------------------------------- |
| 所需线程       | **单线程** (因为仅定义一个loop，所有event均在一个loop中) |
| 编程方式       | 同步                                                     |
| 实现效果       | **异步**                                                 |
| 是否需要锁机制 | 否                                                       |
| 程序级别       | 函数级                                                   |
| 实现机制       | **事件循环＋协程**                                       |
| 总耗时         | 最耗时事件的时间                                         |
| 应用场景       | IO密集型任务等                                           |

