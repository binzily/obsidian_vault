[什么是Docker？看这一篇干货文章就够了！ - 知乎](https://zhuanlan.zhihu.com/p/187505981)

**如何使用docker**

docker中有这样几个概念：

- dockerfile
- image
- container

实际上你可以简单的把image理解为可执行程序，container就是运行起来的进程。

那么写程序需要源代码，那么“写”image就需要dockerfile，dockerfile就是image的源代码，docker就是"编译器"。

因此我们只需要在dockerfile中指定需要哪些程序、依赖什么样的配置，之后把dockerfile交给“编译器”docker进行“编译”，也就是docker build命令，生成的可执行程序就是image，之后就可以运行这个image了，这就是docker run命令，image运行起来后就是docker container。



- Docker Desktop是一个wsl2实例，在G:\DockerImage\DockerDesktopWSL\main里面，主要用于运行 Docker **守护进程**，负责管理容器的生命周期，处理容器的创建、启动、停止等操作。

- Docker Desktop Data又是一个wsl2实例，在G:\DockerImage\DockerDesktopWSL\disk里面，用于**存储 Docker 镜像、容器数据卷**等数据相关内容 。

在Docker Desktop上pull image需要开**全局代理**，然后删除docker镜像。或者**加镜像开热点开规则**代理。

```bash
# docker run -d -it --name ktd --network host -v E:\Work\Develop:/workspace --gpus all -p 7890:7890 pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
docker run -d -it --name rb --network host -v E:\Work:/workspace --gpus all pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
docker exec -it ktd /bin/bash
```

```bash
conda init bash
source /root/.bashrc #这两行执行完后才能用conda的命令
conda create -n robobrain --clone base -y #复制base环境的包（docker Hub里面预装的包）到新环境
```

```bash
# 更新包索引
apt-get update
# 安装 git（可能需要先安装 sudo）
apt-get install -y git
```

```cmd
docker cp E:\Edgedownloads\capeformer-split1-1shot-4c40dfd2_20230713.pth capeformer:/workspace/capeformer/pth/ #执行这个命令前先新建pth目录。
```

