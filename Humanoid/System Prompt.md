0.工作环境为WSL2，conda环境为ThesisMaster，需要什么包你自己可以安装。
1.如果要写项目的代码、思路等，非必要不写try-except这类逻辑，这类代码是冗余的。
2.如果是测试，生产的日志、文件等请在tmp（没有就创建）目录下，不要影响主文件。
3.禁止批量删除文件或目录。
不要使用：
- `del /s`
- `rd /s`
- `rmdir /s`
- `Remove-Item -Recurse`
- `rm -rf`

需要删除文件时，只能一次删除一个明确路径的文件。

正确示例：
Remove-Item "/mnt/c/path/to/file.txt"

如果需要批量删除文件，应停止操作，并询问用户，让用户手动删除。
