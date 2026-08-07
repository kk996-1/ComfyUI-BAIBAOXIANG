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


class KeywordImageBatch:
    MAX_OUTPUT_IMAGES = 9
    MAX_KEYWORDS = 20
    IMAGE_LABELS = ("图一", "图二", "图三", "图四", "图五", "图六", "图七", "图八", "图九")

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        keyword_inputs = {
            f"关键词{i}": ("STRING", {"default": "", "multiline": False})
            for i in range(1, s.MAX_KEYWORDS + 1)
        }
        return {
            "required": {
                "文件夹路径": ("STRING", {"default": "", "multiline": False}),
                **keyword_inputs,
                "自动": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭"}),
                "最大提交数": ("INT", {"default": 10, "min": 1, "max": 500, "step": 1}),
                "序号": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1, "display": "hidden"}),
            },
            "optional": {
                "字符串列表": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    INPUT_IS_LIST = True

    RETURN_TYPES = ("STRING",) + ("IMAGE",) * MAX_OUTPUT_IMAGES
    RETURN_NAMES = ("字符串",) + IMAGE_LABELS
    FUNCTION = "run"
    CATEGORY = "百宝箱/队列"
    
    # 每次运行都强制更新，确保节点状态改变
    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

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

    def _load_image_tensor(self, image_path):
        img = Image.open(image_path)
        img = img.convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        t = torch.from_numpy(arr)
        return t.unsqueeze(0)

    def _find_matching_images(self, folder_path, keywords):
        folder_path = folder_path.strip().strip('"').strip("'")
        folder_path = os.path.expanduser(os.path.expandvars(folder_path))

        if not folder_path or not os.path.isdir(folder_path):
            return {}

        exts = {".png", ".jpg", ".jpeg"}
        keyword_images = {}

        for name in os.listdir(folder_path):
            name_lower = name.lower()
            ext = os.path.splitext(name_lower)[1]
            if ext not in exts:
                continue

            base_name = os.path.splitext(name)[0]
            for keyword in keywords:
                keyword = keyword.strip()
                if not keyword:
                    continue
                if keyword.lower() in base_name.lower():
                    if keyword not in keyword_images:
                        keyword_images[keyword] = os.path.join(folder_path, name)

        return keyword_images

    def _match_keywords_in_text(self, text, keywords):
        matched = []
        text_lower = text.lower()
        for keyword in keywords:
            keyword = keyword.strip()
            if not keyword:
                continue
            if keyword.lower() in text_lower:
                matched.append(keyword)
        return matched

    def _to_prompt_list(self, maybe_batch):
        maybe_batch = self._unwrap_maybe_batch(maybe_batch)
        if maybe_batch is None:
            return []
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

    def _collect_keywords(self, raw_keywords):
        keywords = []
        for keyword in raw_keywords:
            value = self._unwrap_scalar(keyword)
            keywords.append("" if value is None else str(value))
        return keywords

    def run(
        self,
        文件夹路径,
        关键词1,
        关键词2,
        关键词3,
        关键词4,
        关键词5,
        关键词6,
        关键词7,
        关键词8,
        关键词9,
        关键词10,
        关键词11,
        关键词12,
        关键词13,
        关键词14,
        关键词15,
        关键词16,
        关键词17,
        关键词18,
        关键词19,
        关键词20,
        自动,
        最大提交数,
        序号,
        字符串列表=None,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
    ):
        文件夹路径 = self._unwrap_scalar(文件夹路径)
        端口号 = COMFYUI_PORT
        自动 = bool(self._unwrap_scalar(自动))
        最大提交数 = int(self._unwrap_scalar(最大提交数))
        序号 = int(self._unwrap_scalar(序号))
        prompt = self._unwrap_scalar(prompt)
        unique_id = self._unwrap_scalar(unique_id)

        keywords = self._collect_keywords([
            关键词1, 关键词2, 关键词3, 关键词4, 关键词5,
            关键词6, 关键词7, 关键词8, 关键词9, 关键词10,
            关键词11, 关键词12, 关键词13, 关键词14, 关键词15,
            关键词16, 关键词17, 关键词18, 关键词19, 关键词20,
        ])
        prompts = self._to_prompt_list(字符串列表)

        if not prompts:
            raise ValueError("字符串列表 不能为空")

        keyword_images = self._find_matching_images(文件夹路径, keywords)
        total = min(最大提交数, len(prompts))

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

            client_id = "kk_baibaoxiang_keyword_batch"
            for i in range(1, total):
                new_prompt = copy.deepcopy(prompt)
                inputs = new_prompt[node_id].get("inputs", {})
                inputs["序号"] = i
                inputs["自动"] = False
                new_prompt[node_id]["inputs"] = inputs

                self._queue_prompt_http(new_prompt, client_id, 端口号)
                time.sleep(0.5)

        current_text = prompts[idx]
        matched_keywords = self._match_keywords_in_text(current_text, keywords)
        matched_keywords = matched_keywords[:self.MAX_OUTPUT_IMAGES]

        append_str = []
        matched_images = []

        for i, keyword in enumerate(matched_keywords):
            if keyword in keyword_images:
                img_path = keyword_images[keyword]
                img_tensor = self._load_image_tensor(img_path)
                matched_images.append(img_tensor)
                append_str.append(f"{self.IMAGE_LABELS[i]}是{keyword}")

        if append_str:
            out_text = current_text + " " + "，".join(append_str)
        else:
            out_text = current_text

        # 根据匹配的图片数量动态返回，最多9张
        result = [out_text]
        for img in matched_images:
            result.append(img)

        # 补齐到9个图片输出（如果没有匹配到足够的图片）
        while len(result) < self.MAX_OUTPUT_IMAGES + 1:  # 1个字符串 + 9个图片
            result.append(torch.zeros((1, 64, 64, 3), dtype=torch.float32))

        return tuple(result)
