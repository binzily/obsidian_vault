```
conda create -n edmp python=3.8

conda activate edmp

pip install ninja cmake<3.27
```

1. 准备源码并打补丁
```cmd
:: 随意工作目录
mkdir D:\tmp\ikfast && cd D:\tmp\ikfast

:: 只下载源码包，跳过依赖
pip download ikfast_pybind==0.1.2 --no-binary :all: --no-deps

tar -xf ikfast_pybind-0.1.2.tar.gz
cd ikfast_pybind-0.1.2
```

① 修改 setup_cmake_utils.py  
  • 注释掉强行指定 `-A x64` 和 `/m`  
  • Windows 分支留一个 `pass` 占位  
  • 再添加一行，把 **构建类型** 传给 Ninja  

```python
cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]   # ← 新增
```

改之后的 setup_cmake_utils.py：

> ```
> """ CMake util functions for building python extension module with pybind11 & CMake build system
> 
> this script is adapted from cmake_example project for pybind11
> https://github.com/pybind/cmake_example
> 
> for more info for extension modules:
> https://docs.python.org/2/distutils/setupscript.html
> """
> from __future__ import absolute_import, print_function
> 
> import os
> import sys
> import re
> import platform
> import subprocess
> 
> from setuptools import setup, Extension, find_packages
> from setuptools.command.build_ext import build_ext
> from distutils.version import LooseVersion
> 
> 
> class CMakeExtension(Extension):
>     def __init__(self, name, sourcedir=''):
>         Extension.__init__(self, name, sources=[])
>         self.sourcedir = os.path.abspath(sourcedir)
> 
> 
> class CMakeBuild(build_ext):
>     def run(self):
>         try:
>             out = subprocess.check_output(['cmake', '--version'])
>         except OSError:
>             raise RuntimeError("CMake must be installed to build the following extensions: " +
>                                ", ".join(e.name for e in self.extensions))
> 
>         if platform.system() == "Windows":
>             cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
>             if cmake_version < '3.1.0':
>                 raise RuntimeError("CMake >= 3.1.0 is required on Windows")
> 
>         for ext in self.extensions:
>             self.build_extension(ext)
> 
>     def build_extension(self, ext):
>         extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
>         cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
>                       '-DPYTHON_EXECUTABLE=' + sys.executable]
> 
>         cfg = 'Debug' if self.debug else 'Release'
>         build_args = ['--config', cfg]
> 
>         if platform.system() == "Windows":
>             # 把生成的 .pyd 放到 Release/Debug 目录各自下面
>             cmake_args += [
>                 "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir),
> 		"-DCMAKE_BUILD_TYPE=" + cfg
>             ]
> 
>             # 在 64-bit Python 下 Ninja 会自动生成 x64 目标，
>             # 不再需要手动加 -A x64；但 if 语句必须留个内容，用 pass 占位
>             if sys.maxsize > 2 ** 32:
>                 pass
> 
>             # Ninja 自己负责并行，也不需要把 /m 传给 MSBuild
>             # build_args += ["--", "/m"]
> 
>         else:
>             cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
>             build_args += ["--", "-j2"]
> 
>         env = os.environ.copy()
>         env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
>                                                               self.distribution.get_version())
>         if not os.path.exists(self.build_temp):
>             os.makedirs(self.build_temp)
>         subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
>         subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
> ```

② 修改顶层 CMakeLists.txt  

```cmake
if (MSVC)
    add_compile_definitions(
        _CRT_SECURE_NO_WARNINGS
        _CRT_DECLARE_NONSTDC_NAMES=1
        _CRT_INTERNAL_NONSTDC_NAMES=1   # 缺它 _invalid_parameter 会缺失
    )
endif()
```

（pybind11 自带 CMakeLists 里没有 /Za，不需改。）

改之后的CMakeLists.txt：

> ```
> cmake_minimum_required(VERSION 2.8.12)
> project(ikfast_pybind)
> 
> if (MSVC)
>     add_compile_definitions(
>         _CRT_SECURE_NO_WARNINGS
>         _CRT_DECLARE_NONSTDC_NAMES=1
>         _CRT_INTERNAL_NONSTDC_NAMES=1      # ←★ 必加：真正放开 _invalid_parameter
>     )
> endif()
> 
> set(IKFAST_PYBIND_ROOT "${CMAKE_CURRENT_LIST_DIR}")
> set(IKFAST_PYBIND_SOURCE_DIR "${IKFAST_PYBIND_ROOT}/src")
> set(IKFAST_PYBIND_EXTERNAL "${IKFAST_PYBIND_ROOT}/ext")
> 
> add_subdirectory(${IKFAST_PYBIND_EXTERNAL})
> add_subdirectory(${IKFAST_PYBIND_SOURCE_DIR})
> ```

2. 打开 MSVC 环境并编译
```cmd
:: VS-x64 开发者终端
call "D:\visualstudio\VisualStudioIDE\VC\Auxiliary\Build\vcvars64.bat"

:: 让 Conda 的新 CMake 生效
set "PATH=%CONDA_PREFIX%\Scripts;%CONDA_PREFIX%\Library\bin;%PATH%"

:: Ninja 生成器
set CMAKE_GENERATOR=Ninja

:: 清理旧 build
rmdir /s /q build 2>nul

:: 编译安装
pip install -e .
```

若日志中每条 cl.exe 参数都出现  

最后cd到robofin的目录里运行pip install -e .

