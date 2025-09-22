# YOLO目标检测并裁剪图片+OCR识别

## 功能说明

本脚本使用YOLO模型进行目标检测，裁剪出目标区域，并使用OCR技术识别图片中的文字。

## 安装依赖

### 方法一：使用安装脚本（推荐）
1. 双击运行 `install_dependencies.bat` 文件
2. 等待安装完成

### 方法二：手动安装
1. 打开命令行终端
2. 运行以下命令：
   ```
   pip install -r requirements.txt
   ```

## 使用方法

### 基本用法
```bash
python extract_tag.py --model best.pt --input images --output cropped --ocr
```

### 参数说明
- `--model`: YOLO模型路径，默认为 'best.pt'
- `--input`: 输入文件夹，默认为 'images'
- `--output`: 输出文件夹，默认为 'cropped'
- `--compress`: 图片压缩质量，可选值为 'h'(高)、'm'(中)、'l'(低)、'n'(不压缩)，默认为 'n'
- `--target`: 目标类别名称，默认为 'WhiteTag'
- `--ocr`: 是否启用OCR识别，使用此参数启用
- `--ocr_model`: OCR大模型名称，默认为 'gpt-4o'
- `--ocr_key`: OCR API KEY
- `--ocr_provider`: OCR服务商，默认为 'OPENAI'
- `--ocr_batch`: 是否批量OCR推理，使用此参数启用
- `--ocr_proxy`: OCR代理地址
- `--ocr_base_url`: OCR自定义服务商的base_url
- `--ocr_max_retries`: OCR识别失败时的最大重试次数，默认为 3
- `--ocr_system_prompt`: OCR系统提示词模板文件路径
- `--ocr_user_prompt`: OCR用户提示词模板文件路径

### 提示词模板

您可以通过提供提示词模板文件来自定义OCR的提示词：

1. 创建系统提示词模板文件（例如 `system_prompt.txt`）
2. 创建用户提示词模板文件（例如 `user_prompt.txt`）
3. 在运行脚本时指定这些文件：
   ```bash
   python extract_tag.py --ocr --ocr_system_prompt system_prompt.txt --ocr_user_prompt user_prompt.txt
   ```

### 示例

1. 基本OCR识别：
   ```bash
   python extract_tag.py --ocr
   ```

2. 使用自定义API KEY和模型：
   ```bash
   python extract_tag.py --ocr --ocr_key "your_api_key" --ocr_model "gpt-4-vision-preview"
   ```

3. 使用自定义提示词模板：
   ```bash
   python extract_tag.py --ocr --ocr_system_prompt system_prompt.txt --ocr_user_prompt user_prompt.txt
   ```

4. 启用图片压缩和批量处理：
   ```bash
   python extract_tag.py --ocr --compress m --ocr_batch
   ```

## 输出文件

- 裁剪后的图片保存在输出文件夹中
- OCR识别结果保存在输出文件夹的 `ocr_result.txt` 文件中
- 处理日志保存在输出文件夹的 `process.log` 文件中
- 已处理图片的状态记录保存在输出文件夹的 `processed_images.json` 文件中

## 注意事项

1. 确保已安装所有依赖包
2. 确保YOLO模型文件存在
3. 确保输入文件夹中包含图片文件
4. 如果使用OCR功能，确保提供有效的API KEY
5. 网络连接正常，以便调用OCR API

## 故障排除

如果遇到问题，请检查：
1. 依赖包是否正确安装
2. API KEY是否有效
3. 网络连接是否正常
4. 输入图片格式是否支持（支持.jpg, .jpeg, .png, .bmp格式）