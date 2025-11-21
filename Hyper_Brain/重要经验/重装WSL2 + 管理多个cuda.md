linux：

https://blog.csdn.net/qq_42406643/article/details/109545766[理清GPU、CUDA、CUDA Toolkit、cuDNN关系以及下载安装-CSDN博客](https://blog.csdn.net/qq_42406643/article/details/109545766)

[CUDA的正确安装/升级/重装/使用方式 - 知乎](https://zhuanlan.zhihu.com/p/520536351)

windows：这两个博客要结合看，有些东西不需要安装，并且可以撞到D盘。

[CUDA Toolkit安装教程（Windows）-CSDN博客](https://blog.csdn.net/qq_42951560/article/details/116131410)

[(4 封私信 / 4 条消息) 【保姆级】Windows 安装 CUDA 和 cuDNN - 知乎](https://zhuanlan.zhihu.com/p/32400431090)

# 1.重装WSL2

- 得开steam++下载，不然会报错域名解析失败。

wsl --list --online列出可在线获取的 WSL 发行版。

wsl --install -d Ubuntu-20.04安装指定版本的 Ubuntu。

wsl --export Ubuntu G:\Ubuntu\ubuntu.tar导出。

 wsl --unregister Ubuntu

wsl --import Ubuntu G:\Ubuntu G:\ubuntu1\ubuntu.tar --version 2导入。

迁移路径之后如果要用其他用户启动，需要用wsl的命令来指定：

wsl --distribution Ubuntu-22.04 --user bx

在ubuntu中更改默认用户：vim /etc/wsl.conf，加入：

[user]

default=bx

# 2.安装cudatoolkit（runfile）和cudnn（local）

​	CUDA在大版本下向下兼容。比如你装了CUDA11.5，它支持CUDA11.0-CUDA11.5的环境，但不支持CUDA10.2。所以选版本的时候直接选符合你要求的大版本的最新版就行。

**切换CUDA版本实际上就是修改路径中的CUDA目录。**

**如果你只是临时修改，你可以用我上面提到的命令，export PATH=.......来修改；**

**如果你想修改你的用户的默认CUDA，你可以修改~/.bashrc，来自动修改你用户的路径**

gcc --version 查看是否安装gcc，如果没有安装：

```bash
sudo apt-get update
sudo apt-get install build-essential
sudo apt install gcc-11 g++-11
```

但是cudatoolkit11.8不支持gcc11，所以安装gcc9：

```bash
sudo apt install gcc-9 g++-9
#使用 update-alternatives 命令切换 GCC 版本：
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9
#验证当前使用的 GCC 版本
gcc --version
```

- 将 `/usr/local/cuda-11.8/bin` 添加到 `PATH` 环境变量中，这样系统才能找到 CUDA 相关的可执行文件。
- 将 `/usr/local/cuda-11.8/lib64` 添加到 `LD_LIBRARY_PATH` 环境变量，或者把该路径添加到 `/etc/ld.so.conf` 文件中并执行 `ldconfig` 命令，以此保证系统能正确加载 CUDA 库文件。

```bash
export PATH=/usr/local/cuda-11.8/bin:$PATH #这个命令直接让nvcc —V生效，有时候这一句代码就够了下面的export和建立软连接都不需要看。但是软链接更适合多版本cuda管理。
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

​	若要永久添加，可编辑 `~/.bashrc` 或 `/etc/profile` 文件，在文件末尾添加上述 `export` 命令，然后执行 `source ~/.bashrc` 或 `source /etc/profile` 使设置生效。

**当本机上安装有多个版本cuda时可以通过一下步骤进行管理/版本切换：sudo vim ~/.bashrc加入以下的内容：**

```bash
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 #和cudnn有关，这种方式如果LD_LIBRARY_PATH本身是空的话会造成冒号错误，CUDA_HOME同理，正确方式应该像下面：
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
#或者直接export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export PATH=$PATH:/usr/local/cuda/bin
export CUDA_HOME=/usr/local/cuda:$CUDA_HOME
#或者直接export CUDA_HOME=/usr/local/cuda
```

这样是为了对后面的软连接做铺垫。

修改完毕保存，`source ~/.bashrc`

```bash
ls -l /usr/local #查看软连接
sudo rm -rf /usr/local/cuda # 删除旧版本的软连接（这里可能直接删除文件，慎重！！！）
sudo ln -s /usr/local/cuda-9.1 /usr/local/cuda # 建立新版本的软连接
                    						# 前面的路径是需要的版本的cuda的安装路径。
```

**cudatoolkit安装完后安装cudnn：**

cudnn安装的时候就应该和nvcc的cuda匹配，不然是安装不了的。（可能我按照9.8.0的cudnn时，他检测到了我有11.6的cuda，所以还安装了8.3.2的cudnn来适配cuda。）

进入到cudnn下载的安装路径下，命令行输入以下命令进行解压操作：

```shell
tar -xzvf cudnn-10.1-linux-x64-v8.0.5.39.tgz //这里cudnn-10.1-linux-x64-v8.0.5.39.tgz是我们下载的cudnn的压缩包
```

随后在当前路径的命令行终端输入以下三条命令进行cudnn的安装（其实就是把原来的安装目录的文件放到软连接下）：

```shell
sudo cp cuda/include/cudnn.h    /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn*    /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h   /usr/local/cuda/lib64/libcudnn*
```

```
print(torch.__config__.show())
```

[Linux下安装cuda和对应版本的cudnn_linux怎么在自己的环境中安装cuda和cudnn-CSDN博客](https://blog.csdn.net/qq_44961869/article/details/115954258)

# 3. 安装anaconda

安装只有一个ENTER，踩坑。

```bash
# 1. 先 cd 到根目录下
cd
# 2. 下载安装包：在此地址 https://www.anaconda.com/download/success 中找到安装包的链接
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
# 3. 安装 anaconda
bash Anaconda3-2024.02-1-Linux-x86_64.sh
# 4. 按照 anaconda 提示进行安装，默认安装到 /home/用户名/anaconda3
```

```bash
# 1. 打开系统环境变量文件
vim ~/.bashrc
# 2. 添加 Anaconda 环境变量
export PATH="/home/用户名/anaconda3/bin:$PATH"
# 3. （可选）设置 Anaconda 快捷键
alias act='conda activate'
alias deact='conda deactivate'
# 4. 更新环境变量
source ~/.bashrc
# 5. 验证是否添加完成
conda --version
```

# 4.安装flashattn

```shell
pip install flash_attn-2.7.2.post1+cu12torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation
```

