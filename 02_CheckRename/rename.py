#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
集成的图片核查与重命名工具（GUI）。

功能：
- 选择目标文件夹（读取该目录下所有图片文件）
- 在界面中预览图片、输入新文件名、逐张记录/跳转
- 支持加载建议映射文件（old new），会在核查界面显示建议
- 导出重命名映射为文本文件
- 应用重命名到磁盘，处理冲突（将冲突文件移到“重复”子文件夹）

打包说明见 README.md（推荐使用 PyInstaller）。
"""

import os
import shutil
import re
import tkinter as tk
from tkinter import filedialog, messagebox
import time

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')


class CheckRenameApp:
    def __init__(self, root):
        self.root = root
        root.title('图片核查与重命名')
        root.geometry('900x600')

        # state
        self.folder = ''
        self.image_list = []
        self.current_index = 0
        self.rename_map = {}       # old filename -> newname (no ext)
        self.suggestion_map = {}   # old filename -> suggested newname (no ext)
        self.start_time = None
        self.end_time = None

        # layout
        self.create_widgets()

    def create_widgets(self):
        left = tk.Frame(self.root, bg='#fff')
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = tk.Frame(self.root, width=360, padx=12, pady=12)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        # image preview area
        self.canvas = tk.Canvas(left, bg='#f5f5f5')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas_image = None

        # controls on right
        # 上方 IO 控件：选择文件夹、加载映射、导出、应用重命名
        io_frame = tk.Frame(right)
        io_frame.pack(fill=tk.X, pady=(0,8))
        tk.Button(io_frame, text='📁 选择文件夹', command=self.select_folder).grid(row=0, column=0, sticky='ew')
        tk.Button(io_frame, text='📄 加载映射文件', command=self.select_mapping_file).grid(row=0, column=1, sticky='ew')
        tk.Button(io_frame, text='📖 导出映射', command=self.export_map).grid(row=1, column=0, sticky='ew', pady=(6,0))
        tk.Button(io_frame, text='🚀 应用重命名', command=self.apply_renames).grid(row=1, column=1, sticky='ew', pady=(6,0))

        # 文件名与修改输入区域
        tk.Label(right, text='当前文件:', anchor='w').pack(fill=tk.X)
        self.lbl_filename = tk.Label(right, text='', wraplength=320, anchor='w', fg='#333')
        self.lbl_filename.pack(fill=tk.X, pady=(0,8))

        tk.Label(right, text='输入新文件名（无需后缀）:', anchor='w').pack(fill=tk.X)
        self.entry_new = tk.Entry(right, font=('Segoe UI', 12))
        self.entry_new.pack(fill=tk.X, pady=(0,8))

        self.lbl_progress = tk.Label(right, text='当前进度：0/0')
        self.lbl_progress.pack(pady=(0,8))

        btn_frame = tk.Frame(right)
        btn_frame.pack(fill=tk.X, pady=(8,0))

        tk.Button(btn_frame, text='⬆上一个', command=self.prev_image).grid(row=0, column=0, sticky='ew')
        tk.Button(btn_frame, text='下一个⬇', command=self.next_image).grid(row=0, column=1, sticky='ew')
        tk.Button(btn_frame, text='✅ 记录', command=self.record_rename).grid(row=0, column=2, sticky='ew')

        # 正则与自定义替换区域
        repl_frame = tk.LabelFrame(right, text='替换工具', padx=4, pady=4)
        repl_frame.pack(fill=tk.X, pady=(8,0))

        tk.Label(repl_frame, text='规则').grid(row=1, column=0, columnspan=5, sticky='w', pady=(6,0))
        self.custom_map_entry = tk.Entry(repl_frame)
        self.custom_map_entry.grid(row=2, column=0, columnspan=2, sticky='ew', pady=4)
        tk.Button(repl_frame, text='🔄 替换', command=self.apply_custom_map).grid(row=2, column=2, padx=4)

        # keyboard bindings: 上/下 切换
        self.root.bind('<Up>', lambda e: self.prev_image())
        self.root.bind('<Down>', lambda e: self.next_image())
        self.root.bind('<Control-Return>', lambda e: self.record_rename())

    def select_folder(self):
        folder = filedialog.askdirectory(title='选择目标文件夹')
        if not folder:
            return
        self.folder = folder
        # start timing when folder loaded
        self.start_time = time.time()
        self.end_time = None
        self.load_images()

    def load_images(self):
        self.image_list = [f for f in os.listdir(self.folder)
                           if os.path.isfile(os.path.join(self.folder, f)) and f.lower().endswith(IMAGE_EXTENSIONS)]
        self.image_list.sort()
        self.current_index = 0
        self.rename_map = {}
        self.update_ui()

    def select_mapping_file(self):
        fp = filedialog.askopenfilename(title='选择映射文件', filetypes=[('文本文件', '*.txt;*.csv'), ('所有文件', '*.*')])
        if not fp:
            return
        self.load_mapping_file(fp)

    def load_mapping_file(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception:
            with open(path, 'r', encoding='gbk', errors='ignore') as f:
                lines = f.readlines()

        self.suggestion_map.clear()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 支持: old new  或 old,new
            if ',' in line:
                parts = [p.strip() for p in line.split(',') if p.strip()]
            else:
                parts = line.split()

            if len(parts) >= 2:
                old = parts[0]
                new = parts[1]
                # remove extension from suggested name
                new_base = os.path.splitext(new)[0]
                self.suggestion_map[old] = new_base

        messagebox.showinfo('提示', f'已加载映射建议（{len(self.suggestion_map)} 条）')
        self.update_ui()

    def apply_regex_suggestions(self):
        """使用正则表达式对当前文件列表生成建议名称（仅预览/建议，不立即重命名）。"""
        pattern = self.regex_from_entry.get().strip()
        repl = self.regex_to_entry.get()
        if not pattern:
            messagebox.showwarning('警告', '请输入正则表达式模式')
            return
        try:
            cre = re.compile(pattern)
        except re.error as e:
            messagebox.showerror('错误', f'正则表达式无效: {e}')
            return

        # apply to basename (不含后缀)，避免影响扩展名
        count = 0
        for fname in self.image_list:
            base0 = os.path.splitext(fname)[0]
            newbase = cre.sub(repl, base0)
            if newbase and newbase != base0:
                self.suggestion_map[fname] = newbase
                count += 1

        messagebox.showinfo('完成', f'生成 {count} 条正则建议')
        self.update_ui()

    def apply_custom_map(self):
        """从用户输入的自定义映射字符串生成建议。
        格式示例："-正=-1,-侧=-2"
        意味着在每个文件名中替换子串"-正"为"-1"，"-侧"为"-2"。
        """
        s = self.custom_map_entry.get().strip()
        if not s:
            messagebox.showwarning('警告', '请输入自定义映射字符串')
            return

        # 规范化输入：支持全角等号／箭头等变体
        norm = s.replace('＝', '=').replace('→', '->').replace('⇒', '->')
        # 解析多个映射，支持英文逗号/中文逗号/分号分隔
        raw_parts = [p.strip() for p in re.split('[,，;；]+', norm) if p.strip()]
        mappings = []
        for p in raw_parts:
            # 支持几种 key/value 分隔符：= 或 -> 或 =>
            m = re.split('\s*(?:=|->|=>)\s*', p, maxsplit=1)
            if len(m) == 2:
                a, b = m[0].strip(), m[1].strip()
                if a:
                    mappings.append((a, b))

        if not mappings:
            messagebox.showwarning('警告', '未解析到有效的映射对（old=new）')
            return

        count = 0
        match_counts = {a: 0 for a, _ in mappings}
        examples = {a: [] for a, _ in mappings}

        for fname in self.image_list:
            base0 = os.path.splitext(fname)[0]
            newbase = base0
            for a, b in mappings:
                if a and a in newbase:
                    newbase = newbase.replace(a, b)
                    match_counts[a] += 1
                    if len(examples[a]) < 3:
                        examples[a].append(base0)
            if newbase != base0:
                self.suggestion_map[fname] = newbase
                count += 1

        # 构建统计/诊断信息
        if count == 0:
            lines = ['未生成任何建议。诊断信息：']
            if mappings:
                for a, _ in mappings:
                    lines.append(f'映射 "{a}" 匹配到 {match_counts.get(a,0)} 个文件')
                    if examples.get(a):
                        lines.append(' 示例: ' + ', '.join(examples[a]))
            # 显示部分文件名以便检查
            sample = [os.path.splitext(f)[0] for f in self.image_list[:10]]
            if sample:
                lines.append('文件样例: ' + ', '.join(sample))
            messagebox.showinfo('完成', '\n'.join(lines))
        else:
            # 显示总数并简短报告每个映射的命中数
            report = [f'生成 {count} 条自定义替换建议']
            for a in match_counts:
                report.append(f'"{a}" -> {match_counts[a]}')
            messagebox.showinfo('完成', '\n'.join(report))

        self.update_ui()

    def update_ui(self):
        total = len(self.image_list)
        if total == 0:
            self.lbl_filename.config(text='(尚未选择文件夹或文件夹内无图片)')
            self.entry_new.delete(0, tk.END)
            self.lbl_progress.config(text='当前进度：0/0')
            self.canvas.delete('all')
            return

        fname = self.image_list[self.current_index]
        self.lbl_filename.config(text=fname)
        # preset entry: recorded rename or suggestion
        preset = self.rename_map.get(fname) or self.suggestion_map.get(fname) or ''
        self.entry_new.delete(0, tk.END)
        self.entry_new.insert(0, preset)

        self.lbl_progress.config(text=f'当前进度：{len(self.rename_map)}/{total}')
        self.show_image(os.path.join(self.folder, fname))

    def show_image(self, path):
        self.canvas.delete('all')
        if Image is None:
            # Pillow not installed
            self.canvas.create_text(200, 100, text='请安装 Pillow 才能显示预览（pip install pillow）', fill='red')
            return

        try:
            img = Image.open(path)
            # resize to fit canvas
            cw = self.canvas.winfo_width() or 600
            ch = self.canvas.winfo_height() or 400
            img.thumbnail((cw - 20, ch - 20), Image.LANCZOS)
            self.tkimg = ImageTk.PhotoImage(img)
            self.canvas.create_image(cw//2, ch//2, image=self.tkimg, anchor='c')
        except Exception as e:
            self.canvas.create_text(200, 100, text=f'无法打开图片: {e}', fill='red')

    def prev_image(self):
        if not self.image_list:
            return
        self.current_index = (self.current_index - 1) % len(self.image_list)
        self.update_ui()

    def next_image(self):
        if not self.image_list:
            return
        # require non-empty new filename before moving next
        newval = self.entry_new.get().strip()
        if not newval:
            messagebox.showwarning('警告', '请输入新文件名，才可继续到下一张')
            return
        # save and advance
        self._save_current_entry()
        self.current_index = (self.current_index + 1) % len(self.image_list)
        self.update_ui()

    def _save_current_entry(self):
        if not self.image_list:
            return
        fname = self.image_list[self.current_index]
        newbase = self.entry_new.get().strip()
        if newbase:
            self.rename_map[fname] = newbase
            # 检查是否完成全部命名
            self.check_completion()

    def record_rename(self):
        if not self.image_list:
            return
        fname = self.image_list[self.current_index]
        newbase = self.entry_new.get().strip()
        if not newbase:
            messagebox.showwarning('警告', '请输入新文件名')
            return
        self.rename_map[fname] = newbase
        messagebox.showinfo('已记录', f'已记录: {fname} → {newbase}')
        self.lbl_progress.config(text=f'当前进度：{len(self.rename_map)}/{len(self.image_list)}')
        self.check_completion()

    def check_completion(self):
        # 当所有图片都有记录时弹出耗时提示
        if not self.image_list:
            return
        if len(self.rename_map) >= len(self.image_list):
            # avoid double-popup
            if self.end_time is None:
                self.end_time = time.time()
                if self.start_time is None:
                    elapsed = self.end_time - (self.start_time or self.end_time)
                else:
                    elapsed = self.end_time - self.start_time
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                messagebox.showinfo('完成', f'🎉 所有文件已完成！耗时 {mins} 分 {secs} 秒')

    def export_map(self):
        if not self.rename_map:
            messagebox.showwarning('提示', '没有重命名记录')
            return
        fp = filedialog.asksaveasfilename(title='导出映射文件', defaultextension='.txt', filetypes=[('文本文件', '*.txt')])
        if not fp:
            return
        try:
            with open(fp, 'w', encoding='utf-8') as f:
                for old, newbase in self.rename_map.items():
                    ext = os.path.splitext(old)[1]
                    f.write(f'{old} {newbase}{ext}\n')
            messagebox.showinfo('完成', f'已导出到 {fp}')
        except Exception as e:
            messagebox.showerror('错误', f'导出失败: {e}')

    def apply_renames(self):
        if not self.folder:
            messagebox.showwarning('警告', '请先选择目标文件夹')
            return
        if not self.rename_map:
            messagebox.showwarning('提示', '没有重命名记录')
            return

        confirm = messagebox.askyesno('确认', f'将对 {len(self.rename_map)} 个文件执行重命名，是否继续？')
        if not confirm:
            return

        duplicate_dir = os.path.join(self.folder, '重复')
        os.makedirs(duplicate_dir, exist_ok=True)

        success = 0
        for old, newbase in list(self.rename_map.items()):
            old_path = os.path.join(self.folder, old)
            if not os.path.exists(old_path):
                continue
            new_name = newbase + os.path.splitext(old)[1]
            new_path = os.path.join(self.folder, new_name)

            # handle conflict: if target exists, move target to  "重复" folder
            if os.path.exists(new_path):
                try:
                    target_conflict = os.path.join(duplicate_dir, os.path.basename(new_path))
                    if os.path.exists(target_conflict):
                        os.remove(target_conflict)
                    shutil.move(new_path, target_conflict)
                except Exception as e:
                    messagebox.showwarning('警告', f'移动冲突文件失败: {e}')
                    continue

            try:
                os.rename(old_path, new_path)
                success += 1
            except Exception as e:
                messagebox.showwarning('警告', f'重命名失败 {old} → {new_name}: {e}')

        messagebox.showinfo('完成', f'成功重命名 {success} 个文件')
        # reload list
        self.load_images()


def main():
    root = tk.Tk()
    app = CheckRenameApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()