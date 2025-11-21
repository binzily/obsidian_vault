# 网络通讯

![image-20241117114930049](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241117114930049.png)

![image-20241117115543800](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241117115543800.png)

![image-20241117124251623](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241117124251623.png)

![image-20241117124428070](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241117124428070.png)

![image-20241117125454247](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241117125454247.png)

![image-20241117163229777](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241117163229777.png)

<img src="C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241117164633863.png" alt="image-20241117164633863" style="zoom:33%;" />

![image-20241117164850285](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241117164850285.png)

![image-20241117170915947](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241117170915947.png)

![image-20241117171500450](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241117171500450.png)

![image-20241117173438336](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241117173438336.png)











# 一.第一章

## 1.Linux简介



<img src="C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241029112751098.png" alt="image-20241029112751098" style="zoom:67%;" />

​	==bin==文件夹存的常用的exe的命令如（ls，pwd等）。

​	==WSL==可以在本机win11用本机真实硬件构建一个Linux系统，并且可以通过powershell启动Linux。

​	==虚拟机快照==。

​	==蓝色==文件夹，==白色==普通文件，==绿色==可执行文件。

## 2.Linux命令

Linux命令本质上是一个个的二进制可执行程序，和windows的.exe文件是一个意思。

### 1）命令格式

<img src="C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241029170555644.png" alt="image-20241029170555644" style="zoom:40%;" />

### 2）常用命令

①==ls==：列出目录内所有内容

==cd==：change directory。

==~==：home目录。

==pwd==：打印当前工作目录  。

.表示当前目录， ..表示上一级目录。

==mkdir -p== 创建文件夹，==touch==创建文件，==cat==、==more==（支持翻页，按q退出）查看文件内容

==cp==复制文件，==mv==移动文件，==rm==删除文件（*用作模糊匹配 ）。

==which==查找程序文件存放的位置，==find==查找文件。

==grep==从文件中过滤关键字，==wc==统计文件行数和单词数量，==管道符====|==将左边命令的结果作为右边命令的输入。

==echo==在命令行内输出指定内容，==反引号==相当于python的转义字符，重定向符==>==与==>>==,==tail==查看文件尾部内容，跟踪文件的最新更改 。

### 3）用户与用户组

①==su - root==切换root用户（-表示在切换用户后加载环境变量），==sudo==可以为用户临时授权root（需要root用户执行==visodu==命令） 。

②用root账户==groupadd==创建和==groupdel==删除用户组。接着就可以创建用户：==useradd[-g -d]==，-g指定用户的组，-d指定用户的home路径。==userdel[-r]==，-r删除用户home目录。==id[用户名]==，查看用户所在组。==usermod -aG 用户组 用户名==，指定用户加入指定用户/组。

③==chmod [-R] 权限 文件或文件夹==，可以修改文件、文件夹的权限信息，选项-R对文件夹内全部内容采取同样操作。

==chown [-R] 用户[:]用户组 文件或文件夹==，可以修改文件、文件夹

## 3.Linux实用操作

①==ctrl+l==清空，==ctrl+d==退出，==ctrl+c==，==ctrl+a==和==ctrl+e==，==ctrl+ins==和==shift+ins==

②Linux命令行的“应用商店”==yum==。

### 1）tmux命令

==功能==：
    (1) 分屏。
    (2) 允许断开Terminal连接后，继续运行进程。
结构：
    一个tmux可以包含多个session，一个session可以包含多个window，一个window可以包含多个pane。
    实例：
        tmux:
            session 0:
                window 0:
                    pane 0
                    pane 1
                    pane 2
                    ...
                window 1
                window 2
                ...
            session 1
            session 2
            ...
操作：
    (1) tmux：新建一个session，其中包含一个window，window中包含一个pane，pane里打开了一个shell对话框。
    (2) 按下Ctrl + a后手指松开，然后按==%==：将当前pane左右平分成两个pane。
    (3) 按下Ctrl + a后手指松开，然后按=="==（注意是双引号"）：将当前pane上下平分成两个pane。
    (4) Ctrl + d：关闭当前pane；如果当前window的所有pane均已关闭，则自动关闭window；如果当前session的所有window均已关闭，则自动关闭session。
    (5) 鼠标点击可以选pane。
    (6) 按下ctrl + a后手指松开，然后按方向键：选择相邻的pane。
    (7) 鼠标拖动pane之间的分割线，可以调整分割线的位置。
    (8) 按住ctrl + a的同时按方向键，可以调整pane之间分割线的位置。
    (9) 按下ctrl + a后手指松开，然后按==z==：将当前pane全屏/取消全屏。
    (10) 按下ctrl + a后手指松开，然后按==d==：挂起当前session。
    (11) ==tmux a==：打开之前挂起的session。
    (12) 按下ctrl + a后手指松开，然后按==s==：选择其它session。
        方向键 —— 上：选择上一项 session/window/pane
        方向键 —— 下：选择下一项 session/window/pane
        方向键 —— 右：展开当前项 session/window
        方向键 —— 左：闭合当前项 session/window
    (13) 按下Ctrl + a后手指松开，然后按==c==：在当前session中创建一个新的window。
    (14) 按下Ctrl + a后手指松开，然后按==w==：选择其他window，操作方法与(12)完全相同。
    (15) 按下Ctrl + a后手指松开，然后按==PageUp==：翻阅当前pane内的内容。
    (16) 鼠标滚轮：翻阅当前pane内的内容。
    (17) 在tmux中选中文本时，需要按住shift键。（仅支持Windows和Linux，不支持Mac，不过该操作并不是必须的，因此影响不大）
    (18) tmux中复制/粘贴文本的通用方式：
        (1) 按下Ctrl + a后松开手指，然后按==[==
        (2) 用鼠标选中文本，被选中的文本会被自动复制到tmux的剪贴板
        (3) 按下Ctrl + a后松开手指，然后按==]==，会将剪贴板中的内容粘贴到光标处

​    (19)==tmux kill-server==关闭所有session

### 2）vim命令

<img src="C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241029231615276.png" alt="image-20241029231615276" style="zoom:45%;" />

功能：
    (1) 命令行模式下的文本编辑器。
    (2) 根据文件扩展名自动判别编程语言。支持代码缩进、代码高亮等功能。
    (3) 使用方式：vim filename
        如果已有该文件，则打开它。
        如果没有该文件，则打开个一个新的文件，并命名为filename
模式：
    (1) 一般命令模式
        默认模式。命令输入方式：类似于打游戏放技能，按不同字符，即可进行不同操作。可以复制、粘贴、删除文本等。
    (2) 编辑模式
        在一般命令模式里按下i，会进入编辑模式。
        按下ESC会退出编辑模式，返回到一般命令模式。
    (3) 命令行模式
        在一般命令模式里按下==:/?==三个字母中的任意一个，会进入命令行模式。命令行在最下面。
        可以查找、替换、保存、退出、配置编辑器等。
操作：
    (1) i：进入编辑模式
    (2) ESC：进入一般命令模式
    (3) h 或 左箭头键：光标向左移动一个字符
    (4) j 或 向下箭头：光标向下移动一个字符
    (5) k 或 向上箭头：光标向上移动一个字符
    (6) l 或 向右箭头：光标向右移动一个字符
    (7) ==n<Space>==：n表示数字，按下数字后再按空格，光标会向右移动这一行的n个字符
    (8) 0 或 功能键[==Home==]：光标移动到本行开头
    (9) \$或功能键[==End==]：光标移动到本行末尾
    (10) G：光标移动到最后一行
    (11) ==:n== 或 ==nG==：n为数字，光标移动到第n行
    (12) gg：光标移动到第一行，相当于1G
    (13) ==n<Enter>==：n为数字，光标向下移动n行
    (14) ==/word==：向光标之下寻找第一个值为word的字符串。
    (15) ==?word==：向光标之上寻找第一个值为word的字符串。
    (16) n：重复前一个查找操作
    (17) N：反向重复前一个查找操作
    (18) ==:n1,n2s/word1/word2/g：==n1与n2为数字，在第n1行与n2行之间寻找word1这个字符串，并将该字符串替换为word2
    (19) ==:1,\$s/word1/word2/g：==将全文的word1替换为word2
    (20) ==:1,$s/word1/word2/gc：==将全文的word1替换为word2，且在替换前要求用户确认。
    (21) ==v==：选中文本
    (22) ==d==：删除选中的文本
    (23) ==dd==: 删除当前行
    (24) ==y==：复制选中的文本
    (25) ==yy==: 复制当前行
    (26) ==p==: 将复制的数据在光标的下一行/下一个位置粘贴
    (27) ==u==：撤销
    (28) Ctrl + r：取消撤销
    (29) ==大于号 >==：将选中的文本整体向右缩进一次
    (30) ==小于号 <==：将选中的文本整体向左缩进一次
    (31) :w 保存
    (32) :w! 强制保存
    (33) :q 退出
    (34) :q! 强制退出
    (35) :wq 保存并退出
    (36) :set paste 设置成粘贴模式，取消代码自动缩进
    (37) :set nopaste 取消粘贴模式，开启代码自动缩进
    (38) :set nu 显示行号
    (39) :set nonu 隐藏行号
    (40) gg=G：将全文代码格式化
    (41) :noh 关闭查找关键词高亮
    (42) ==Ctrl + q==：当vim卡死时，可以取消当前正在执行的命令
异常处理：
    每次用vim编辑文件时，会自动创建一个==.filename.swp==的临时文件。
    如果打开某个文件时，该文件的swp文件已存在，则会报错。此时解决办法有两种：
        (1) 找到正在打开该文件的程序，并退出which 
        (2) 直接删掉该swp文件即可

## 4.Shell语法

shell是我们与操作系统沟通的语言，例如==AC Terminal中的命令行可以看成是一个“shell脚本在逐行执行”==，bash是Linux的==脚本解释器==。

### 1）

