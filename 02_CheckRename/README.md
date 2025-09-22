# 图片核查与批量重命名工具

这是一个基于 Tkinter 的桌面工具，用于逐张核查图片并记录重命名建议，支持：

- 选择目标文件夹并预览目录内的图片
- 在界面中为每张图片输入（或采用建议）新文件名（不含后缀）
- 从文本映射文件加载建议（每行：old new，支持空格或逗号分隔）
- 导出已记录的映射（old new.ext）到文本文件
- 将映射应用到磁盘，遇到目标已存在时，会把目标文件移动到子目录 `重复/`

打包为单文件 exe（Windows）

1. 建议先创建虚拟环境并安装依赖：

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. 使用 PyInstaller 打包（推荐安装在虚拟环境中）：

```powershell
pyinstaller --onefile --windowed rename.py
```

生成的 exe 在 `dist\rename.exe`。

依赖

- Pillow（用于图片预览，非必须，但推荐）

注意

- 脚本会基于文件名（不含路径）进行匹配与重命名，请确保选择的是包含目标图片的文件夹。
- 打包后请在目标系统上测试应用（尤其是权限与编码）。
