1.huggingface-cli隶属于huggingface_hub库，不仅可以下载模型、数据，还可以可以登录huggingface、上传模型、数据等。

**创建新环境**：

创建新环境要先装python，不然pip安装的东西都安装到D盘anaconda的Lib下了。

**修改环境变量：**

![image-20250112094223404](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20250112094223404.png)

**安装依赖**：

```python
pip install -U huggingface_hub
```

**登录**：

```python
huggingface-cli login
```

**输入Access Token：**

在huggingface官网的设置里面。

**使用cli命令下载：**

```python
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ./llama-3.1-8b-instruct
```

**huggingface从没下载完的地方继续下载**：

```python
 D:\LM\Stage1>huggingface-cli download --resume-download meta-llama/Llama-3.1-8B-Instruct --local-dir ./llama3.1_weights --resume
```

