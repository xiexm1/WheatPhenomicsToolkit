#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ultralytics import YOLO
from openai import OpenAI
import cv2
import os
import argparse
from PIL import Image
import numpy as np
import base64
import httpx
import time
import json
import hashlib

def get_quality_settings(quality):
    """根据质量等级返回图片压缩参数配置"""
    settings = {
        "h": {"max_size": 1920, "quality": 90, "compress_level": 1},
        "m": {"max_size": 1080, "quality": 75, "compress_level": 3},
        "l": {"max_size": 720, "quality": 50, "compress_level": 6}
    }
    return settings.get(quality, settings["m"])

def compress_image(image, settings):
    """压缩图片"""
    # 转换为PIL Image对象
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # 等比例缩放
    img.thumbnail((settings["max_size"], settings["max_size"]), Image.LANCZOS)
    
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def get_image_hash(image_path):
    """获取图片文件的哈希值作为唯一标识"""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_prompt_template(file_path):
    """加载提示词模板文件"""
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"警告: 无法读取提示词模板文件 {file_path}: {str(e)}")
    return None

def load_processed_images(status_file_path):
    """加载已处理图片的状态记录"""
    if os.path.exists(status_file_path):
        try:
            with open(status_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_processed_image(status_file_path, image_path, output_path, ocr_result=None):
    """保存已处理图片的状态记录"""
    status_data = load_processed_images(status_file_path)
    image_hash = get_image_hash(image_path)
    status_data[image_path] = {
        'hash': image_hash,
        'output_path': output_path,
        'ocr_result': ocr_result,
        'processed_at': time.time()
    }
    with open(status_file_path, 'w', encoding='utf-8') as f:
        json.dump(status_data, f, ensure_ascii=False, indent=2)

# OCR模型API调用类（支持多种服务商）
class OCRModel:
    def __init__(self, provider, api_key, model_name, system_prompt=None, user_prompt=None, max_tokens=1000, proxy=None, base_url=None):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt or "你是一个OCR助手，请根据要求将图片文字提取为纯文本。"
        self.user_prompt = user_prompt or "提取格式：红牌是“NY-”+数字，中间白牌是数字1-5，最后为“正/侧”。三部分用“-”连接，去除所有空格，仅输出结果。"
        self.max_tokens = max_tokens
        self.proxy = proxy
        self.base_url = base_url
        self.client = None
        self._init_client()

    def _init_client(self):
        """根据服务商初始化客户端"""
        try:
            if self.provider.upper() == 'OPENAI':
                if self.proxy:
                    self.client = OpenAI(
                        http_client=httpx.Client(
                            transport=httpx.HTTPTransport(proxy=self.proxy)
                        )
                    )
                else:
                    self.client = OpenAI()
            elif self.provider.upper() == 'ALI':
                base_url = self.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1/"
                if self.proxy:
                    self.client = OpenAI(
                        base_url=base_url,
                        http_client=httpx.Client(
                            transport=httpx.HTTPTransport(proxy=self.proxy)
                        )
                    )
                else:
                    self.client = OpenAI(base_url=base_url)
            elif self.provider.upper() == 'SELF':
                if not self.base_url:
                    raise ValueError("自定义服务商需要提供base_url")
                if self.proxy:
                    self.client = OpenAI(
                        base_url=self.base_url,
                        api_key=self.api_key,
                        http_client=httpx.Client(
                            transport=httpx.HTTPTransport(proxy=self.proxy)
                        )
                    )
                else:
                    self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            else:
                raise ValueError(f"不支持的服务商: {self.provider}")
            
            # 设置API Key
            if self.api_key:
                if self.provider.upper() == 'OPENAI':
                    # OpenAI客户端会自动从环境变量读取API Key
                    os.environ['OPENAI_API_KEY'] = self.api_key
                else:
                    # 其他服务商需要手动设置API Key
                    self.client.api_key = self.api_key
        except Exception as e:
            print(f"初始化客户端失败: {str(e)}")
            self.client = None

    def encode_image(self, image):
        """图像编码为base64"""
        # image: numpy array (BGR)
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        from io import BytesIO
        buf = BytesIO()
        img_pil.save(buf, format='PNG')
        img_bytes = buf.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')

    def ocr_with_client(self, image, max_retries=3, image_hash=None):
        """使用OpenAI客户端进行OCR识别，支持重试"""
        if not self.client:
            return "[ERROR] OCR客户端未初始化，请检查API密钥设置"
        
        last_error = None
        for attempt in range(max_retries + 1):  # +1 是因为第一次尝试不算重试
            try:
                base64_img = f"data:image/png;base64,{self.encode_image(image)}"
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": self.system_prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.user_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"{base64_img}"}
                                }
                            ],
                        }
                    ],
                    max_tokens=self.max_tokens,
                )
                
                # 检查响应是否有效
                if not response or not hasattr(response, 'choices') or not response.choices:
                    raise ValueError("OCR响应为空或格式不正确")
                
                # 检查choices[0]是否存在
                if not response.choices[0] or not hasattr(response.choices[0], 'message'):
                    raise ValueError("响应中不包含有效的消息内容")
                    
                # 检查message.content是否存在
                if not hasattr(response.choices[0].message, 'content') or not response.choices[0].message.content:
                    raise ValueError("响应中不包含文本内容")
                
                # 如果成功，返回结果
                return response.choices[0].message.content
                
            except Exception as e:
                last_error = e
                if attempt < max_retries:  # 如果不是最后一次尝试，则等待后重试
                    wait_time = 2 ** attempt  # 指数退避：1s, 2s, 4s...
                    hash_info = f" [{image_hash[:8]}...]" if image_hash else ""
                    print(f"OCR识别失败{hash_info}，{wait_time}秒后重试 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                    time.sleep(wait_time)
                else:
                    hash_info = f" [图片Hash: {image_hash[:8]}...]" if image_hash else ""
                    print(f"OCR识别最终失败{hash_info}，已达到最大重试次数 {max_retries}: {str(e)}")
        
        # 所有重试都失败后返回错误信息
        hash_info = f" [图片Hash: {image_hash[:8]}...]" if image_hash else ""
        error_msg = f"[ERROR] OCR识别失败{hash_info}（已重试{max_retries}次）: {str(last_error)}"
        return error_msg

    def ocr_batch_with_client(self, images, max_retries=3, image_hashes=None):
        """批量图片识别，逐张调用，支持重试"""
        results = []
        for i, img in enumerate(images):
            img_hash = image_hashes[i] if image_hashes and i < len(image_hashes) else None
            results.append(self.ocr_with_client(img, max_retries, img_hash))
            time.sleep(1) # 防止QPS过高
        return results

    def ocr(self, images, batch=False, max_retries=3, image_hashes=None):
        """根据服务商选择OCR方法"""
        if self.client:
            if batch:
                return self.ocr_batch_with_client(images, max_retries, image_hashes)
            else:
                if not images or len(images) == 0:
                    return ["[ERROR] 没有提供图片进行OCR识别"]
                img_hash = image_hashes[0] if image_hashes and len(image_hashes) > 0 else None
                return [self.ocr_with_client(images[0], max_retries, img_hash)]
        else:
            return [f"[ERROR] {self.provider} 客户端初始化失败"]

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='YOLO目标检测并裁剪图片+OCR识别')
    parser.add_argument('--model', type=str, default='best.pt', help='模型路径')
    parser.add_argument('--input', type=str, default='images', help='输入文件夹')
    parser.add_argument('--output', type=str, default='cropped', help='输出文件夹')
    parser.add_argument('--compress', type=str, choices=['h', 'm', 'l', 'n'], default='n', help='图片压缩质量(h/m/l)或不压缩(n)')
    parser.add_argument('--target', type=str, default='WhiteTag', help='目标类别名称(默认: WhiteTag)')
    parser.add_argument('--ocr', action='store_true', help='是否启用OCR识别')
    parser.add_argument('--ocr_model', type=str, default='gpt-4o', help='OCR大模型名称')
    parser.add_argument('--ocr_key', type=str, default='', help='OCR API KEY')
    parser.add_argument('--ocr_provider', type=str, default='OPENAI', help='OCR服务商')
    parser.add_argument('--ocr_batch', action='store_true', help='是否批量OCR推理')
    parser.add_argument('--ocr_proxy', type=str, default=None, help='OCR代理地址')
    parser.add_argument('--ocr_base_url', type=str, default=None, help='OCR自定义服务商的base_url')
    parser.add_argument('--ocr_max_retries', type=int, default=3, help='OCR识别失败时的最大重试次数')
    parser.add_argument('--ocr_system_prompt', type=str, default=None, help='OCR系统提示词模板文件路径')
    parser.add_argument('--ocr_user_prompt', type=str, default=None, help='OCR用户提示词模板文件路径')
    args = parser.parse_args()

    # 设置输入输出路径
    input_folder = args.input
    output_folder = args.output

    # 创建输出目录（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 日志文件路径
    log_path = os.path.join(output_folder, 'process.log')
    log_file = open(log_path, 'w', encoding='utf-8')
    ocr_result_path = os.path.join(output_folder, 'ocr_result.txt') if args.ocr else None
    
    # 状态记录文件路径
    status_file_path = os.path.join(output_folder, 'processed_images.json')
    processed_images = load_processed_images(status_file_path)

    #加载模型
    model = YOLO(args.model)

    # 加载提示词模板
    system_prompt = load_prompt_template(args.ocr_system_prompt)
    user_prompt = load_prompt_template(args.ocr_user_prompt)
    
    # 初始化OCR模型
    ocr_model = None
    if args.ocr:
        ocr_model = OCRModel(
            provider=args.ocr_provider,
            api_key=args.ocr_key,
            model_name=args.ocr_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            proxy=args.ocr_proxy,
            base_url=args.ocr_base_url
        )
        if ocr_model.client is None:
            print("警告: OCR模型初始化失败，将跳过OCR处理")
            log_file.write("警告: OCR模型初始化失败，将跳过OCR处理\n")

    # 获取压缩设置
    compress_settings = get_quality_settings(args.compress) if args.compress != 'n' else None

    # 获取目标类别的 class_id
    target_class_name = args.target
    class_names = model.names  # 获取模型的类别名称字典 {id: name}
    class_id = None
    for cid, name in class_names.items():
        if name == target_class_name:
            class_id = cid
            break

    if class_id is None:
        log_file.write(f"Class '{target_class_name}' not found in model classes.\n")
        log_file.close()
        raise ValueError(f"Class '{target_class_name}' not found in model classes.")

    # 遍历图像文件夹
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    img_list = [img_file for img_file in os.listdir(input_folder) if img_file.lower().endswith(image_exts)]
    total = len(img_list)
    cropped_imgs = []
    cropped_img_names = []
    # 创建一个映射，将裁剪后的图片路径映射到原始图片路径
    cropped_to_original = {}
    for idx, img_file in enumerate(img_list, 1):
        image_path = os.path.join(input_folder, img_file)
        
        # 检查图片是否已经处理过
        current_hash = get_image_hash(image_path)
        if image_path in processed_images and processed_images[image_path]['hash'] == current_hash:
            print(f"[{idx}/{total}] 跳过已处理: {img_file}")
            log_file.write(f"Skipping already processed: {image_path}\n")
            
            # 如果需要OCR且之前没有OCR结果，则只进行OCR
            if args.ocr and 'ocr_result' not in processed_images[image_path]:
                output_path = processed_images[image_path]['output_path']
                if os.path.exists(output_path):
                    cropped_img = cv2.imread(output_path)
                    if cropped_img is not None:
                        print(f"[{idx}/{total}] 仅进行OCR识别: {img_file}")
                        log_file.write(f"OCR only for: {image_path}\n")
                        
                        # 检查OCR模型是否可用
                        if ocr_model and ocr_model.client:
                            try:
                                # 获取图片hash
                                img_hash = get_image_hash(image_path)
                                ocr_result_list = ocr_model.ocr([cropped_img], batch=False, max_retries=args.ocr_max_retries, image_hashes=[img_hash])
                                if ocr_result_list and len(ocr_result_list) > 0:
                                    ocr_result = ocr_result_list[0]
                                    save_processed_image(status_file_path, image_path, output_path, ocr_result)
                                    
                                    # 记录OCR结果
                                    with open(ocr_result_path, 'a', encoding='utf-8') as ocr_out:
                                        log_file.write(f"OCR for {output_path}:\n{ocr_result}\n\n")
                                        # 对于单独处理的图片，直接使用原始文件名和结果格式
                                        original_filename = os.path.basename(image_path)
                                        ocr_out.write(f"{original_filename},{ocr_result}\n")
                                else:
                                    error_msg = "[ERROR] OCR返回结果为空"
                                    log_file.write(f"OCR处理失败: {error_msg}\n")
                                    print(f"OCR处理失败: {error_msg}")
                            except Exception as e:
                                error_msg = f"OCR处理失败: {str(e)}"
                                log_file.write(f"{error_msg}\n")
                                print(f"{error_msg}")
                        else:
                            log_file.write("OCR模型未正确初始化，跳过OCR处理\n")
                            print("OCR模型未正确初始化，跳过OCR处理")
            continue
            
        print(f"[{idx}/{total}] 正在处理: {img_file}")
        log_file.write(f"Processing: {image_path}\n")

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            log_file.write(f"Failed to load image: {image_path}\n")
            continue

        # 推理
        results = model(image)

        # 初始化边界框的最小和最大坐标
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')

        # 遍历结果
        for i, r in enumerate(results):
            boxes = r.boxes.cpu().numpy()
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id != class_id:
                    continue

                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 更新最小外接矩形的边界
                min_x = min(min_x, x1)
                min_y = min(min_y, y1)
                max_x = max(max_x, x2)
                max_y = max(max_y, y2)

        # 如果没有检测到目标，跳过该图像
        if min_x == float('inf') or min_y == float('inf'):
            log_file.write(f"No {target_class_name} detected in {image_path}. Skipping.\n")
            continue

        # 裁剪图像
        cropped_img = image[min_y:max_y, min_x:max_x]

        # 如果启用压缩,先压缩裁剪后的图像
        if compress_settings:
            cropped_img = compress_image(cropped_img, compress_settings)
            log_file.write(f"Compressed image with quality: {args.compress}\n")

        # 构建保存路径（保持同名，但不同文件夹）
        base_name = os.path.splitext(img_file)[0]
        ext = os.path.splitext(img_file)[1]
        suffix = '_cropped' if not compress_settings else f'_cropped_{args.compress}'
        save_path = os.path.join(output_folder, f"{base_name}{suffix}{ext}")

        # 保存处理后的图像
        if compress_settings and ext.lower() in ['.jpg', '.jpeg']:
            # 对于JPEG格式,使用PIL保存以应用压缩质量设置
            Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)).save(
                save_path, 
                quality=compress_settings['quality'], 
                optimize=True
            )
        else:
            # 其他格式使用cv2保存
            cv2.imwrite(save_path, cropped_img)
        
        log_file.write(f"Saved image: {save_path}\n")
        
        # 保存处理状态（不包括OCR结果）
        save_processed_image(status_file_path, image_path, save_path)

        # 收集待OCR的图片
        if args.ocr:
            cropped_imgs.append(cropped_img)
            cropped_img_names.append(save_path)
            # 添加映射关系：裁剪后的图片路径 -> 原始图片路径
            cropped_to_original[save_path] = image_path

    # OCR识别
    if args.ocr and ocr_model and cropped_imgs:
        log_file.write(f"\n[OCR] 开始识别... 共 {len(cropped_imgs)} 张图片\n")
        try:
            # 收集所有图片的hash
            image_hashes = []
            for fname in cropped_img_names:
                # 从映射中获取原始图片路径
                original_path = cropped_to_original.get(fname, fname)
                img_hash = get_image_hash(original_path)
                image_hashes.append(img_hash)
            
            if args.ocr_batch:
                log_file.write("[OCR] 使用批量处理模式\n")
                ocr_results = ocr_model.ocr(cropped_imgs, batch=True, max_retries=args.ocr_max_retries, image_hashes=image_hashes)
            else:
                log_file.write("[OCR] 使用单张处理模式\n")
                ocr_results = []
                for i, (img, fname) in enumerate(zip(cropped_imgs, cropped_img_names)):
                    log_file.write(f"[OCR] 正在处理第 {i+1}/{len(cropped_imgs)} 张图片: {fname}\n")
                    try:
                        ocr_result_list = ocr_model.ocr([img], batch=False, max_retries=args.ocr_max_retries, image_hashes=[image_hashes[i]])
                        if ocr_result_list and len(ocr_result_list) > 0:
                            result = ocr_result_list[0]
                            ocr_results.append(result)
                            log_file.write(f"[OCR] 第 {i+1} 张图片处理完成\n")
                        else:
                            error_msg = "[ERROR] OCR返回结果为空"
                            ocr_results.append(error_msg)
                            log_file.write(f"[OCR] 第 {i+1} 张图片处理失败: {error_msg}\n")
                    except Exception as e:
                        error_msg = f"[ERROR] OCR处理失败: {str(e)}"
                        log_file.write(f"[OCR] 第 {i+1} 张图片处理失败: {error_msg}\n")
                        ocr_results.append(error_msg)
            
            # 检查是否所有OCR都成功（没有错误信息）
            all_success = all(not result.startswith('[ERROR]') for result in ocr_results)
            
            # 使用追加模式打开OCR结果文件，以保留之前的结果
            with open(ocr_result_path, 'a', encoding='utf-8') as ocr_out:
                for fname, result in zip(cropped_img_names, ocr_results):
                    log_file.write(f"OCR for {fname}:\n{result}\n\n")
                    
                    # 如果所有OCR都成功，则按照"yolo提取前的文件名,识别结果"格式输出
                    if all_success:
                        # 获取原始图片文件名（不含路径）
                        original_path = cropped_to_original.get(fname, fname)
                        original_filename = os.path.basename(original_path)
                        ocr_out.write(f"{original_filename},{result}\n")
                    else:
                        # 如果有失败的OCR，则保持原有格式
                        ocr_out.write(f"{fname}:\n{result}\n\n")
                    
                    # 更新状态记录，添加OCR结果
                    # 查找对应的原始图片路径
                    for img_path, status in processed_images.items():
                        if status.get('output_path') == fname:
                            save_processed_image(status_file_path, img_path, fname, result)
                            break
        except Exception as e:
            log_file.write(f"[OCR] 批量处理过程中发生错误: {str(e)}\n")
            # 尝试单张处理
            log_file.write("[OCR] 尝试单张处理模式...\n")
            ocr_results = []
            # 收集所有图片的hash（如果之前没有收集过）
            if 'image_hashes' not in locals():
                image_hashes = []
                for fname in cropped_img_names:
                    # 从映射中获取原始图片路径
                    original_path = cropped_to_original.get(fname, fname)
                    img_hash = get_image_hash(original_path)
                    image_hashes.append(img_hash)
            
            for i, (img, fname) in enumerate(zip(cropped_imgs, cropped_img_names)):
                log_file.write(f"[OCR] 正在处理第 {i+1}/{len(cropped_imgs)} 张图片: {fname}\n")
                try:
                    ocr_result_list = ocr_model.ocr([img], batch=False, max_retries=args.ocr_max_retries, image_hashes=[image_hashes[i]])
                    if ocr_result_list and len(ocr_result_list) > 0:
                        result = ocr_result_list[0]
                        ocr_results.append(result)
                        log_file.write(f"[OCR] 第 {i+1} 张图片处理完成\n")
                    else:
                        error_msg = "[ERROR] OCR返回结果为空"
                        ocr_results.append(error_msg)
                        log_file.write(f"[OCR] 第 {i+1} 张图片处理失败: {error_msg}\n")
                except Exception as e:
                    error_msg = f"[ERROR] OCR处理失败: {str(e)}"
                    log_file.write(f"[OCR] 第 {i+1} 张图片处理失败: {error_msg}\n")
                    ocr_results.append(error_msg)
            
            # 检查是否所有OCR都成功（没有错误信息）
            all_success = all(not result.startswith('[ERROR]') for result in ocr_results)
            
            # 保存单张处理的结果
            with open(ocr_result_path, 'a', encoding='utf-8') as ocr_out:
                for fname, result in zip(cropped_img_names, ocr_results):
                    log_file.write(f"OCR for {fname}:\n{result}\n\n")
                    
                    # 如果所有OCR都成功，则按照"yolo提取前的文件名,识别结果"格式输出
                    if all_success:
                        # 获取原始图片文件名（不含路径）
                        original_path = cropped_to_original.get(fname, fname)
                        original_filename = os.path.basename(original_path)
                        ocr_out.write(f"{original_filename},{result}\n")
                    else:
                        # 如果有失败的OCR，则保持原有格式
                        ocr_out.write(f"{fname}:\n{result}\n\n")
                    
                    # 更新状态记录，添加OCR结果
                    for img_path, status in processed_images.items():
                        if status.get('output_path') == fname:
                            save_processed_image(status_file_path, img_path, fname, result)
                            break

    log_file.close()

if __name__ == "__main__":
    main()
