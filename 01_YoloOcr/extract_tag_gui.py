#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import os
import time
import base64
import json
import hashlib
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from openai import OpenAI
import httpx
import json
import hashlib

# ---------- 复用的核心函数 ----------

def get_quality_settings(quality):
    settings = {
        "h": {"max_size": 1920, "quality": 90, "compress_level": 1},
        "m": {"max_size": 1080, "quality": 75, "compress_level": 3},
        "l": {"max_size": 720, "quality": 50, "compress_level": 6}
    }
    return settings.get(quality, settings["m"])

def compress_image(image, settings):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img.thumbnail((settings["max_size"], settings["max_size"]), Image.LANCZOS)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def get_image_hash(image_path):
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

def load_prompt_template(file_path):
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception:
            return None
    return None

def load_processed_images(status_file_path):
    if os.path.exists(status_file_path):
        try:
            with open(status_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_processed_image(status_file_path, image_path, output_path, ocr_result=None):
    status = load_processed_images(status_file_path)
    img_hash = None
    try:
        with open(image_path, 'rb') as f:
            img_hash = hashlib.md5(f.read()).hexdigest()
    except Exception:
        img_hash = None
    status[image_path] = {
        'hash': img_hash,
        'output_path': output_path,
        'ocr_result': ocr_result,
        'processed_at': time.time()
    }
    try:
        with open(status_file_path, 'w', encoding='utf-8') as f:
            json.dump(status, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ---------- OCRModel 类 (简化) ----------
class OCRModel:
    def __init__(self, provider, api_key, model_name, system_prompt=None, user_prompt=None, max_tokens=1000, proxy=None, base_url=None, logger=None):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt or "你是一个OCR助手，请将图片内容转为需要的文本格式。"
        self.user_prompt = user_prompt or "提取格式：红牌是“NY-”+数字，中间白牌是数字1-5，最后为“正/侧”。三部分用“-”连接，去除所有空格，仅输出结果。"
        self.max_tokens = max_tokens
        self.proxy = proxy
        self.base_url = base_url
        # logger: callable(msg:str) -> None
        self.logger = logger
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
            if self.logger:
                self.logger(f"初始化客户端失败: {str(e)}")
            else:
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
                if self.logger:
                    try:
                        self.logger(f"✅ [{image_hash[:8]}] OCR 成功")
                    except Exception:
                        pass
                return response.choices[0].message.content
                
            except Exception as e:
                last_error = e
                if attempt < max_retries:  # 如果不是最后一次尝试，则等待后重试
                    wait_time = 2 ** attempt  # 指数退避：1s, 2s, 4s...
                    hash_info = f" [{image_hash[:8]}...]" if image_hash else ""
                    msg = f"❎{hash_info}，{wait_time}秒后重试 (尝试 {attempt + 1}/{max_retries}): {str(e)}"
                    if self.logger:
                        try:
                            self.logger(msg)
                        except Exception:
                            pass
                    else:
                        print(msg)
                    time.sleep(wait_time)
                else:
                    hash_info = f" [图片Hash: {image_hash[:8]}...]" if image_hash else ""
                    msg = f"OCR识别最终失败{hash_info}，已达到最大重试次数 {max_retries}: {str(e)}"
                    if self.logger:
                        try:
                            self.logger(msg)
                        except Exception:
                            pass
                    else:
                        print(msg)
        
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

# ---------- GUI 主体 ----------
class ExtractTagGUI:
    def __init__(self, root):
        self.root = root
        root.title('Extract Tag GUI')
        self.create_widgets()
        self.worker_thread = None
        self.stop_flag = threading.Event()

    def create_widgets(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # Model path
        ttk.Label(frm, text='YOLO 模型:').grid(row=0, column=0, sticky='w')
        self.model_var = tk.StringVar(value='tag.pt')
        ttk.Entry(frm, textvariable=self.model_var, width=50).grid(row=0, column=1, sticky='we')
        ttk.Button(frm, text='浏览', command=self.browse_model).grid(row=0, column=2)

        # Input folder
        ttk.Label(frm, text='输入文件夹:').grid(row=1, column=0, sticky='w')
        self.input_var = tk.StringVar(value='images')
        ttk.Entry(frm, textvariable=self.input_var, width=50).grid(row=1, column=1, sticky='we')
        ttk.Button(frm, text='浏览', command=self.browse_input).grid(row=1, column=2)

        # Output folder
        ttk.Label(frm, text='输出文件夹:').grid(row=2, column=0, sticky='w')
        self.output_var = tk.StringVar(value='cropped')
        ttk.Entry(frm, textvariable=self.output_var, width=50).grid(row=2, column=1, sticky='we')
        ttk.Button(frm, text='浏览', command=self.browse_output).grid(row=2, column=2)

        # Compress
        ttk.Label(frm, text='压缩质量:').grid(row=3, column=0, sticky='w')
        self.compress_var = tk.StringVar(value='l')
        ttk.Combobox(frm, textvariable=self.compress_var, values=['h','m','l','n'], width=6).grid(row=3, column=1, sticky='w')

        # Target class
        ttk.Label(frm, text='目标类别:').grid(row=4, column=0, sticky='w')
        self.target_var = tk.StringVar(value='WhiteTag')
        ttk.Entry(frm, textvariable=self.target_var, width=20).grid(row=4, column=1, sticky='w')

        # OCR options
        self.ocr_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text='启用 OCR', variable=self.ocr_var).grid(row=5, column=0, sticky='w')
        ttk.Label(frm, text='OCR 模型:').grid(row=5, column=1, sticky='w')
        self.ocr_model_var = tk.StringVar(value='gpt-4.1-mini')
        ttk.Entry(frm, textvariable=self.ocr_model_var, width=20).grid(row=5, column=1, sticky='e')
        ttk.Label(frm, text='API KEY:').grid(row=6, column=0, sticky='w')
        self.ocr_key_var = tk.StringVar(value='')
        ttk.Entry(frm, textvariable=self.ocr_key_var, width=50, show='*').grid(row=6, column=1, sticky='we')
        ttk.Label(frm, text='API provider:').grid(row=7, column=0, sticky='w')
        self.ocr_provider_var = tk.StringVar(value='SELF')
        ttk.Entry(frm, textvariable=self.ocr_provider_var, width=20).grid(row=7, column=1, sticky='w')
        # 不再支持批量OCR，始终采用逐张稳定处理
        
        # OCR 自定义 Base URL 和 代理
        ttk.Label(frm, text='API Base URL:').grid(row=8, column=0, sticky='w')
        self.ocr_base_url_var = tk.StringVar(value='')
        ttk.Entry(frm, textvariable=self.ocr_base_url_var, width=50).grid(row=8, column=1, sticky='we')
        ttk.Label(frm, text='API Proxy:').grid(row=9, column=0, sticky='w')
        self.ocr_proxy_var = tk.StringVar(value='')
        ttk.Entry(frm, textvariable=self.ocr_proxy_var, width=50).grid(row=9, column=1, sticky='we')

        # Batch OCR options
        self.ocr_batch_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text='启用 批量 OCR', variable=self.ocr_batch_var).grid(row=9, column=2, sticky='w')
        ttk.Label(frm, text='批量大小:').grid(row=10, column=2, sticky='w')
        self.ocr_batch_size_var = tk.IntVar(value=5)
        ttk.Spinbox(frm, from_=1, to=50, textvariable=self.ocr_batch_size_var, width=5).grid(row=10, column=2, sticky='e')

        # Start / Stop
        # OCR prompt & retries
        ttk.Label(frm, text='System Prompt:').grid(row=10, column=0, sticky='nw')
        self.system_prompt_text = tk.Text(frm, height=4, width=60)
        self.system_prompt_text.insert('1.0', '你是一个OCR助手，请将图片内容转为需要的文本格式。')
        self.system_prompt_text.grid(row=10, column=1, columnspan=2, sticky='we')

        ttk.Label(frm, text='User Prompt:').grid(row=11, column=0, sticky='nw')
        self.user_prompt_text = tk.Text(frm, height=4, width=60)
        self.user_prompt_text.insert('1.0', '提取格式：红牌是“NY-”+数字，中间白牌是数字1-5，最后为“正/侧”。三部分用“-”连接，去除所有空格，仅输出结果。')
        self.user_prompt_text.grid(row=11, column=1, columnspan=2, sticky='we')

        ttk.Label(frm, text='OCR 重试次数:').grid(row=12, column=0, sticky='w')
        self.ocr_retries_var = tk.IntVar(value=3)
        ttk.Spinbox(frm, from_=0, to=10, textvariable=self.ocr_retries_var, width=5).grid(row=12, column=1, sticky='w')

        # Start / Stop
        self.start_btn = ttk.Button(frm, text='开始', command=self.start)
        self.start_btn.grid(row=13, column=0, pady=8)
        self.stop_btn = ttk.Button(frm, text='停止', command=self.stop, state=tk.DISABLED)
        self.stop_btn.grid(row=13, column=1, pady=8)

        # Log area
        self.log_text = tk.Text(frm, height=20)
        self.log_text.grid(row=14, column=0, columnspan=3, sticky='nsew')
        frm.rowconfigure(14, weight=1)
        frm.columnconfigure(1, weight=1)

    # Browse helpers
    def browse_model(self):
        p = filedialog.askopenfilename(filetypes=[('pt','*.pt'),('All','*.*')])
        if p: self.model_var.set(p)
    def browse_input(self):
        p = filedialog.askdirectory()
        if p: self.input_var.set(p)
    def browse_output(self):
        p = filedialog.askdirectory()
        if p: self.output_var.set(p)

    def log(self, msg):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        self.log_text.insert(tk.END, f'[{ts}] {msg}\n')
        self.log_text.see(tk.END)

    def start(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo('信息','任务已在运行')
            return
        # prepare args
        params = {
            'model': self.model_var.get(),
            'input': self.input_var.get(),
            'output': self.output_var.get(),
            'compress': self.compress_var.get(),
            'target': self.target_var.get(),
            'ocr': self.ocr_var.get(),
            'ocr_model': self.ocr_model_var.get(),
            'ocr_key': self.ocr_key_var.get(),
            'ocr_provider': self.ocr_provider_var.get(),
            'ocr_base_url': self.ocr_base_url_var.get(),
            'ocr_proxy': self.ocr_proxy_var.get()
        }
        # include prompts and retries
        params['system_prompt'] = self.system_prompt_text.get('1.0', tk.END).strip()
        params['user_prompt'] = self.user_prompt_text.get('1.0', tk.END).strip()
        params['ocr_retries'] = int(self.ocr_retries_var.get())
        self.stop_flag.clear()
        self.worker_thread = threading.Thread(target=self.run_process, args=(params,))
        self.worker_thread.start()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

    def stop(self):
        self.stop_flag.set()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.log('停止请求已发送，等待线程结束...')

    def run_process(self, params):
        try:
            model_path = params['model']
            input_folder = params['input']
            output_folder = params['output']
            compress = params['compress']
            target = params['target']
            do_ocr = params['ocr']

            os.makedirs(output_folder, exist_ok=True)
            # 不再写入日志文件，所有信息直接显示在 GUI
            self.log(f'加载模型 {model_path} ...')
            model = YOLO(model_path)
            self.log(f'加载模型 {model_path}')

            compress_settings = get_quality_settings(compress) if compress != 'n' else None

            # OCR 初始化
            ocr_model = None
            if do_ocr:
                ocr_model = OCRModel(provider=params['ocr_provider'], api_key=params['ocr_key'], model_name=params['ocr_model'], system_prompt=params.get('system_prompt'), user_prompt=params.get('user_prompt'), proxy=params.get('ocr_proxy'), base_url=params.get('ocr_base_url'), logger=self.log)
                if not ocr_model.client:
                    self.log('OCR 初始化失败，OCR 将被跳过')
                    do_ocr = False

            image_exts = ('.jpg','.jpeg','.png','.bmp')
            img_files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_exts)]
            total = len(img_files)

            cropped_imgs = []
            cropped_names = []
            mapping = {}

            # load processed images state
            status_file = os.path.join(output_folder, 'processed_images.json')
            processed_images = load_processed_images(status_file)

            for idx, fname in enumerate(img_files, 1):
                if self.stop_flag.is_set():
                    self.log('收到停止信号，退出处理')
                    break
                self.log(f'[{idx}/{total}] 处理 {fname}')

                img_path = os.path.join(input_folder, fname)
                # skip already processed if hash matches; if OCR missing, schedule OCR-only
                cur_hash = get_image_hash(img_path)
                if img_path in processed_images and processed_images[img_path].get('hash') == cur_hash:
                    prev = processed_images[img_path]
                    if do_ocr and not prev.get('ocr_result'):
                        # load the existing cropped image and schedule for OCR
                        out_path = prev.get('output_path')
                        if out_path and os.path.exists(out_path):
                            cropped = cv2.imread(out_path)
                            if cropped is not None:
                                cropped_imgs.append(cropped)
                                cropped_names.append(out_path)
                                mapping[out_path] = img_path
                                self.log(f'[{idx}/{total}] 已裁剪但无OCR，加入OCR队列: {fname}')
                    else:
                        self.log(f'[{idx}/{total}] 跳过已处理: {fname}')
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    self.log(f'无法读取: {img_path}')
                    continue
                results = model(img)
                min_x,min_y=float('inf'),float('inf')
                max_x,max_y=float('-inf'),float('-inf')
                for r in results:
                    boxes = r.boxes.cpu().numpy()
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        name = model.names.get(cls_id, str(cls_id))
                        if name != target:
                            continue
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        min_x=min(min_x,x1); min_y=min(min_y,y1)
                        max_x=max(max_x,x2); max_y=max(max_y,y2)

                if min_x==float('inf'):
                    self.log(f'未检测到目标: {img_path}')
                    continue
                cropped = img[min_y:max_y, min_x:max_x]
                if compress_settings:
                    cropped = compress_image(cropped, compress_settings)
                    self.log('已压缩图片')
                base,ext=os.path.splitext(fname)
                suffix = '_cropped' if not compress_settings else f'_cropped_{compress}'
                out_path = os.path.join(output_folder, f"{base}{suffix}{ext}")
                if compress_settings and ext.lower() in ['.jpg','jpeg']:
                    Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)).save(out_path, quality=compress_settings['quality'], optimize=True)
                else:
                    cv2.imwrite(out_path, cropped)
                self.log(f'Saved: {out_path}')

                # 保存处理状态（先保存无 OCR 结果）
                try:
                    save_processed_image(status_file, img_path, out_path, None)
                except Exception:
                    pass

                cropped_imgs.append(cropped)
                cropped_names.append(out_path)
                mapping[out_path]=img_path

            # OCR 阶段（在所有图片处理完成后执行一次）
            if do_ocr and ocr_model and cropped_imgs:
                self.log('[OCR] 开始识别')
                try:
                    image_hashes = [get_image_hash(mapping[p]) for p in cropped_names]
                    # 支持批量 OCR：按 user 选择的 batch_size 分批提交
                    if self.ocr_batch_var.get():
                        batch_size = int(self.ocr_batch_size_var.get())
                        for i in range(0, len(cropped_imgs), batch_size):
                            batch_imgs = cropped_imgs[i:i+batch_size]
                            batch_hashes = image_hashes[i:i+batch_size]
                            self.log(f'[OCR] 提交批次 {i//batch_size + 1}, 大小 {len(batch_imgs)}')
                            res_list = ocr_model.ocr(batch_imgs, batch=True, max_retries=params.get('ocr_retries', 3), image_hashes=batch_hashes)
                            for j, result_text in enumerate(res_list):
                                idx = i + j
                                orig = mapping.get(cropped_names[idx], cropped_names[idx])
                                self.log(f'OCR for {os.path.basename(orig)}: {result_text}')
                                try:
                                    save_processed_image(status_file, orig, cropped_names[idx], result_text)
                                except Exception:
                                    pass
                    else:
                        # 逐张稳定处理 OCR
                        for i, img in enumerate(cropped_imgs):
                            res_list = ocr_model.ocr([img], batch=False, max_retries=params.get('ocr_retries', 3), image_hashes=[image_hashes[i]])
                            result_text = res_list[0] if res_list else '[ERROR] OCR返回空'
                            orig = mapping.get(cropped_names[i], cropped_names[i])
                            self.log(f'OCR for {os.path.basename(orig)}: {result_text}')
                            try:
                                save_processed_image(status_file, orig, cropped_names[i], result_text)
                            except Exception:
                                pass
                    self.log('[OCR] 完成')
                except Exception as e:
                    self.log(f'[OCR] 错误: {e}')
                finally:
                    # 清空队列，避免重复处理
                    cropped_imgs.clear()
                    cropped_names.clear()
                    mapping.clear()

            self.log('处理完成')
        except Exception as e:
            self.log(f'异常: {e}')
        finally:
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

# ---------- 主程序 ----------
if __name__ == '__main__':
    root = tk.Tk()
    root.geometry('900x600')
    app = ExtractTagGUI(root)
    root.mainloop()
