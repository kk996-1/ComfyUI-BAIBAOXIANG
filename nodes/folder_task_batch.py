import copy
import json
import ntpath
import os
import re
import time
import urllib.request

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence


try:
    from comfy.cli_args import args
    COMFYUI_PORT = args.port
except Exception:
    COMFYUI_PORT = 8188


class FolderTaskBatch:
    IMAGE_OUTPUTS = 7
    AUDIO_OUTPUTS = 3
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
    AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".aac", ".opus", ".wma", ".aif", ".aiff"}

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "路径输入": ("STRING", {"default": "", "multiline": False}),
                "自动提交": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭"}),
                "序号": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1, "display": "hidden"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE",) * IMAGE_OUTPUTS + ("STRING",) + ("AUDIO",) * AUDIO_OUTPUTS + ("STRING",)
    RETURN_NAMES = (
        "图片1", "图片2", "图片3", "图片4", "图片5", "图片6", "图片7",
        "字符串",
        "音频1", "音频2", "音频3",
        "调试日志",
    )
    FUNCTION = "run"
    CATEGORY = "百宝箱/队列"

    def _natural_key(self, value):
        return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value)]

    def _resolve_task_root(self, relative_path):
        if relative_path is None:
            relative_path = ""
        raw_path = str(relative_path).strip().strip('"').strip("'")
        input_root = os.path.realpath(self._get_input_directory())

        if not raw_path:
            return input_root

        # Accept either slash style so a workflow can move between Windows and Linux.
        portable_path = raw_path.replace("\\", "/")
        if os.path.isabs(raw_path) or ntpath.isabs(raw_path) or portable_path.startswith("/"):
            # Older workflow files may have saved the full input path from another
            # computer, such as /root/comfyui/ComfyUI/input/B.  Keep only the
            # part below input and resolve it against this ComfyUI instance.
            parts = [part for part in portable_path.split("/") if part and part != "."]
            input_index = next(
                (index for index in range(len(parts) - 1, -1, -1) if parts[index].casefold() == "input"),
                None,
            )
            if input_index is None:
                raise ValueError(
                    "绝对路径无法映射到当前 ComfyUI input 目录；请填写 input 目录下的相对路径"
                )
            portable_path = "/".join(parts[input_index + 1:])

        candidate = os.path.realpath(os.path.join(input_root, portable_path))
        try:
            common_root = os.path.commonpath([input_root, candidate])
        except ValueError as exc:
            raise ValueError("路径输入不在 ComfyUI input 目录下") from exc
        if os.path.normcase(common_root) != os.path.normcase(input_root):
            raise ValueError("路径输入不能跳出 ComfyUI input 目录")
        return candidate

    def _get_input_directory(self):
        import folder_paths

        return folder_paths.get_input_directory()

    def _collect_task_folders(self, root):
        if not os.path.isdir(root):
            raise ValueError(f"任务根文件夹不存在: {root}")

        folders = []
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if os.path.isdir(path):
                folders.append(path)
        return sorted(folders, key=lambda path: self._natural_key(os.path.basename(path)))

    def _classify_files(self, folder_path):
        images = []
        audios = []
        texts = []

        for name in os.listdir(folder_path):
            path = os.path.join(folder_path, name)
            if not os.path.isfile(path):
                continue
            extension = os.path.splitext(name)[1].casefold()
            if extension in self.IMAGE_EXTENSIONS:
                images.append((name, path))
            elif extension in self.AUDIO_EXTENSIONS:
                audios.append((name, path))
            elif extension == ".txt":
                texts.append((name, path))

        return images, audios, texts

    def _numbered_files(self, files, prefix, maximum, label):
        numbered = {}
        pattern = re.compile(rf"^{re.escape(prefix)}([1-{maximum}])$")
        for name, path in files:
            stem = os.path.splitext(name)[0]
            match = pattern.fullmatch(stem)
            if match is None:
                raise ValueError(f"{label}文件名不符合 {prefix}1~{prefix}{maximum}: {name}")
            number = int(match.group(1))
            if number in numbered:
                raise ValueError(f"{label}编号重复: {prefix}{number}")
            numbered[number] = path

        if set(numbered) != set(range(1, len(numbered) + 1)):
            expected = ", ".join(f"{prefix}{index}" for index in range(1, len(numbered) + 1))
            raise ValueError(f"{label}编号必须从1连续排列，期望: {expected}")
        return [numbered[index] for index in range(1, len(numbered) + 1)]

    def _validate_task_folder(self, folder_path):
        images, audios, texts = self._classify_files(folder_path)

        if not 1 <= len(images) <= self.IMAGE_OUTPUTS:
            raise ValueError(f"图片数量为{len(images)}，必须是1~{self.IMAGE_OUTPUTS}张")
        if len(texts) != 1:
            raise ValueError(f"txt文本数量为{len(texts)}，必须正好1个")
        if len(audios) > self.AUDIO_OUTPUTS:
            raise ValueError(f"音频数量为{len(audios)}，最多{self.AUDIO_OUTPUTS}个")

        image_paths = self._numbered_files(images, "图片", self.IMAGE_OUTPUTS, "图片")
        audio_paths = self._numbered_files(audios, "音频", self.AUDIO_OUTPUTS, "音频") if audios else []
        return {
            "folder": folder_path,
            "images": image_paths,
            "text": texts[0][1],
            "audios": audio_paths,
        }

    def _scan_tasks(self, root):
        all_folders = self._collect_task_folders(root)
        valid_tasks = []
        invalid_details = []
        for folder_path in all_folders:
            try:
                valid_tasks.append(self._validate_task_folder(folder_path))
            except ValueError as exc:
                invalid_details.append(f"{os.path.basename(folder_path)}: {exc}")

        return all_folders, valid_tasks, invalid_details

    def _read_text(self, text_path):
        try:
            with open(text_path, "r", encoding="utf-8-sig") as text_file:
                return text_file.read()
        except UnicodeDecodeError:
            with open(text_path, "r", encoding="gb18030") as text_file:
                return text_file.read()

    def _load_image(self, image_path):
        with Image.open(image_path) as source:
            output_images = []
            width = None
            height = None
            for frame in ImageSequence.Iterator(source):
                frame = ImageOps.exif_transpose(frame)
                if frame.mode == "I":
                    frame = frame.point(lambda value: value * (1 / 255))
                image = frame.convert("RGB")
                if not output_images:
                    width, height = image.size
                if image.size != (width, height):
                    continue
                pixels = np.array(image).astype(np.float32) / 255.0
                output_images.append(torch.from_numpy(pixels)[None,])
                if source.format == "MPO":
                    break

        if not output_images:
            raise ValueError(f"图片无法读取: {image_path}")
        return torch.cat(output_images, dim=0) if len(output_images) > 1 else output_images[0]

    def _load_audio(self, audio_path):
        # Reuse ComfyUI's decoder so WAV/MP3/FLAC/OGG and other supported formats
        # have the same AUDIO structure as the built-in Load Audio node.
        from comfy_extras.nodes_audio import load

        waveform, sample_rate = load(audio_path)
        return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}

    def _empty_image(self):
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

    def _empty_audio(self):
        return {
            "waveform": torch.zeros((1, 1, 1), dtype=torch.float32),
            "sample_rate": 44100,
        }

    def _queue_prompt_http(self, prompt, client_id):
        url = f"http://127.0.0.1:{COMFYUI_PORT}/prompt"
        payload = {"prompt": prompt, "client_id": client_id}
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()

    def _queue_remaining_tasks(self, prompt, unique_id, total):
        if not isinstance(prompt, dict) or unique_id is None:
            raise ValueError("自动提交需要获取当前 workflow 的 prompt 和 unique_id")

        node_id = str(unique_id)
        if node_id not in prompt:
            raise ValueError(f"未在 prompt 中找到当前节点: {node_id}")

        client_id = "kk_baibaoxiang_folder_task_batch"
        for index in range(1, total):
            new_prompt = copy.deepcopy(prompt)
            node_inputs = new_prompt[node_id].setdefault("inputs", {})
            node_inputs["序号"] = index
            node_inputs["自动提交"] = False
            new_prompt[node_id]["inputs"] = node_inputs
            self._queue_prompt_http(new_prompt, client_id)
            time.sleep(0.15)

    def _debug_log(self, root, all_folders, valid_tasks, invalid_details, current_index, queued):
        lines = [
            f"任务根路径: {root}",
            f"检查文件夹数量: {len(all_folders)}",
            f"符合要求数量: {len(valid_tasks)}",
            f"不符合要求数量: {len(invalid_details)}",
            f"当前任务: {current_index + 1}/{len(valid_tasks)}" if valid_tasks else "当前任务: 无",
            f"自动提交队列: 已提交{queued}个后续任务" if queued else "自动提交队列: 未提交",
        ]
        if invalid_details:
            lines.append("不符合详情:")
            lines.extend(invalid_details[:20])
            if len(invalid_details) > 20:
                lines.append(f"其余{len(invalid_details) - 20}个不符合文件夹未显示")
        return "\n".join(lines)

    def run(self, 路径输入, 自动提交, 序号, prompt=None, unique_id=None):
        root = self._resolve_task_root(路径输入)
        all_folders, valid_tasks, invalid_details = self._scan_tasks(root)

        if not valid_tasks:
            debug = self._debug_log(root, all_folders, valid_tasks, invalid_details, 0, 0)
            result = [self._empty_image() for _ in range(self.IMAGE_OUTPUTS)]
            result.append("")
            result.extend(self._empty_audio() for _ in range(self.AUDIO_OUTPUTS))
            result.append(debug)
            return tuple(result)

        index = int(序号)
        if index < 0 or index >= len(valid_tasks):
            raise ValueError(f"序号超出范围: {index}，当前有效任务范围为0~{len(valid_tasks) - 1}")

        queued = 0
        if bool(自动提交) and index == 0:
            self._queue_remaining_tasks(prompt, unique_id, len(valid_tasks))
            queued = max(0, len(valid_tasks) - 1)

        task = valid_tasks[index]
        images = [self._load_image(path) for path in task["images"]]
        audios = [self._load_audio(path) for path in task["audios"]]
        text = self._read_text(task["text"])
        debug = self._debug_log(root, all_folders, valid_tasks, invalid_details, index, queued)

        result = images[:self.IMAGE_OUTPUTS]
        result.extend(self._empty_image() for _ in range(self.IMAGE_OUTPUTS - len(result)))
        result.append(text)
        result.extend(audios[:self.AUDIO_OUTPUTS])
        result.extend(self._empty_audio() for _ in range(self.AUDIO_OUTPUTS - len(audios)))
        result.append(debug)
        return tuple(result)
