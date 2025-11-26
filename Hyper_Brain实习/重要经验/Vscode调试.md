# Vscode调试python项目

工具只是其中一种方式，远程debug可选pycharm或者vscode, 本文主要是记录一下vscode怎么进行debug，后续会补充remote ssh的方法

参考视频
1【vscode调试深度学习项目全网最细致教程（持续更新）】 https://www.bilibili.com/video/BV1vDakeDE2n/?share_source=copy_web&vd_source=ca4e02b93a63f6b61653cafdecdfc29f
2【nlp开发利器——vscode debug nlp大工程（最最最优雅的方式）】 https://www.bilibili.com/video/BV1wt421V718/?share_source=copy_web&vd_source=ca4e02b93a63f6b61653cafdecdfc29f
3【VSCode Debug Python项目 |  Debug技巧】 https://www.bilibili.com/video/BV1i4421Z7aM/?share_source=copy_web&vd_source=ca4e02b93a63f6b61653cafdecdfc29f

 **最优雅的方式** 

 **安装** 

1安装包 **pip install debugpy -U**

2安装vscode关于python的相关插件

 写配置 

一般情况下，大家都是使用deepspeed、torchrun运行代码。参数都特别多，然后都是使用sh xxxx.sh启动脚本。

 在python代码里面（最前面加上这句话） 

```python
import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass
```

 在vscode的launch.json的configuration里面，加上这个配置 

```json
{
    "name": "sh_file_debug",
    "type": "debugpy",
    "request": "attach",
    "connect": {
        "host": "localhost",
        "port": 9501
    }
},
```

🚨 上面的端口号都写一样。别搞错了。

 **启动** 

1.就正常启动，直接sh xxx.sh

2.在你需要debug的python文件，打上debug断点。

3.你看打印出来的东西，是不是出现Waiting for debugger attach.一般来说，都很快，就出现了。

4.再在vscode的debug页面，选择sh_file_debug进行debug。

5.就基本上完成了。确实是很方便。

6.**debug结束之后，别忘记把代码里面的 添加的代码，注销掉**。



https://www.yuque.com/nulinulizainuli-rhgcd/gt6csv/reg1f9q1y69sppwu?singleDoc# 《2. [![img](G:\software\Typora\Typora_files\Hyper_Brain\Vscode调试.assets\9f3ad0659e84c96a711b88dd33f4bc2e945045e0.png)Vscode调试python](https://search.bilibili.com/all?from_source=webcommentline_search&keyword=Vscode调试python&seid=15238424208657696597)项目》

[远程连接服务器](https://www.yuque.com/nulinulizainuli-rhgcd/gt6csv/ithde8p0zvservpp?singleDoc# 《远程连接服务器》)



  