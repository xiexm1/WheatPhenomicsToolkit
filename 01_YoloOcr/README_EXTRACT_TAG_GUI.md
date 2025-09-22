# Extract Tag GUI

这个小工具提供一个 GUI 前端，整合了 YOLO 裁剪、可选图片压缩与调用大模型进行 OCR 识别的功能。

要求
- Windows (已测试)
- Python 3.9+

安装依赖

```powershell
python -m pip install -r gui_requirements.txt
```

运行

```powershell
python extract_tag_gui.py
```

打包为 exe（推荐使用 pyinstaller）

```powershell
pyinstaller --onefile --noconsole --add-data "path\to\best.pt;." extract_tag_gui.py
```

注意事项
- 如果使用自定义服务商或自签名证书，请参考代码中 OCRModel 的 `proxy` 和 `base_url` 参数。
- 打包后若 OpenAI 客户端报错，请确保 `openai` 版本与运行时环境一致。

