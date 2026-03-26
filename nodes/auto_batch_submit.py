import copy
import json
import os
import re
import time
import urllib.request

import numpy as np
import torch
from PIL import Image

# 自动获取 ComfyUI 端口
try:
    from comfy.cli_args import args
    COMFYUI_PORT = args.port
except Exception:
    COMFYUI_PORT = 8188


class AutoBatchSubmit:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "文件夹选择": ("STRING", {"default": "", "multiline": False}),
                "自动": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭"}),
                "最大提交数": ("INT", {"default": 10, "min": 1, "max": 500, "step": 1}),
                "序号": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1, "display": "hidden"}),
            },
            "optional": {
                "字符串批次": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    INPUT_IS_LIST = True

    RETURN_TYPES = ("STRING", "IMAGE", "INT")
    RETURN_NAMES = ("字符串", "图片", "输出编号")
    FUNCTION = "run"
    CATEGORY = "百宝箱/队列"

    def _natural_key(self, s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

    def _unwrap_scalar(self, v):
        if isinstance(v, list):
            return v[0] if len(v) > 0 else None
        return v

    def _unwrap_maybe_batch(self, v):
        if isinstance(v, list) and len(v) == 1 and isinstance(v[0], list):
            return v[0]
        return v

    def _list_images(self, folder_path):
        folder_path = self._unwrap_scalar(folder_path)
        if isinstance(folder_path, str):
            folder_path = folder_path.strip().strip('"').strip("'")
            folder_path = os.path.expanduser(os.path.expandvars(folder_path))

        if not folder_path or not isinstance(folder_path, str):
            return [], ""
        if not os.path.isdir(folder_path):
            raise ValueError(f"文件夹不存在: {folder_path}")

        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        files = []
        for name in os.listdir(folder_path):
            ext = os.path.splitext(name)[1].lower()
            if ext in exts:
                files.append(os.path.join(folder_path, name))

        files.sort(key=lambda p: self._natural_key(os.path.basename(p)))
        return files, folder_path

    def _load_image_tensor(self, image_path):
        img = Image.open(image_path)
        img = img.convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        t = torch.from_numpy(arr)
        return t.unsqueeze(0)

    def _to_prompt_list(self, maybe_batch):
        maybe_batch = self._unwrap_maybe_batch(maybe_batch)
        if maybe_batch is None:
            return None
        if isinstance(maybe_batch, list):
            return ["" if v is None else str(v) for v in maybe_batch]
        return [str(maybe_batch)]

    def _queue_prompt_http(self, new_prompt, client_id, port):
        url = f"http://127.0.0.1:{port}/prompt"
        payload = {"prompt": new_prompt, "client_id": client_id}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()

    def run(self, 文件夹选择, 自动, 最大提交数, 序号, 字符串批次=None, prompt=None, extra_pnginfo=None, unique_id=None):
        文件夹选择 = self._unwrap_scalar(文件夹选择)
        端口号 = COMFYUI_PORT
        自动 = bool(self._unwrap_scalar(自动))
        最大提交数 = int(self._unwrap_scalar(最大提交数))
        序号 = int(self._unwrap_scalar(序号))
        prompt = self._unwrap_scalar(prompt)
        unique_id = self._unwrap_scalar(unique_id)

        prompts = self._to_prompt_list(字符串批次)

        image_files, folder_path = self._list_images(文件夹选择)
        has_images = len(image_files) > 0
        has_prompts = prompts is not None and len(prompts) > 0
        if not has_images and not has_prompts:
            raise ValueError("文件夹路径 和 字符串批次 至少需要提供一个")
        if folder_path and not has_images:
            raise ValueError(f"文件夹内未找到图片: {folder_path}")

        total = int(最大提交数)
        if has_images:
            total = min(total, len(image_files))
        if has_prompts:
            total = min(total, len(prompts))

        idx = int(序号)
        if idx < 0:
            idx = 0
        if idx >= total:
            raise ValueError(f"序号超出范围: {idx}，可用范围 0~{total - 1}")

        if 自动 and idx == 0:
            if not isinstance(prompt, dict) or unique_id is None:
                raise ValueError("自动模式需要获取当前 workflow 的 prompt/unique_id")

            node_id = str(unique_id)
            if node_id not in prompt:
                raise ValueError(f"未在 prompt 中找到当前节点: {node_id}")

            client_id = "kk_baibaoxiang_auto_batch"
            for i in range(1, total):
                new_prompt = copy.deepcopy(prompt)
                inputs = new_prompt[node_id].get("inputs", {})
                inputs["序号"] = i
                inputs["自动"] = False
                new_prompt[node_id]["inputs"] = inputs

                self._queue_prompt_http(new_prompt, client_id, 端口号)
                time.sleep(0.5)

        out_text = "" if not has_prompts else prompts[idx]
        out_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32) if not has_images else self._load_image_tensor(image_files[idx])
        out_index = idx + 1
        return (out_text, out_img, out_index)
