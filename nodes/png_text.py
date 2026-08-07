import binascii
import hashlib
import io
import os
import struct
import zlib

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
TEXT_CHUNK = b"tEXt"
ZTEXT_CHUNK = b"zTXt"
ITEXT_CHUNK = b"iTXt"
TEXT_CHUNKS = {TEXT_CHUNK, ZTEXT_CHUNK, ITEXT_CHUNK}
IEND_CHUNK = b"IEND"
PNG_FRAMES_ATTRIBUTE = "_baibaoxiang_png_frames"


def _encode_text_key(value):
    key = str(value)
    if not key:
        raise ValueError("文本键不能为空")

    try:
        encoded = key.encode("latin-1", "strict")
    except UnicodeEncodeError as exc:
        raise ValueError("文本键只能包含 Latin-1 字符") from exc

    if len(encoded) > 79:
        raise ValueError("文本键长度不能超过79个字节")
    if b"\x00" in encoded:
        raise ValueError("文本键不能包含空字符")
    return encoded


def _build_chunk(chunk_type, payload):
    chunk_data = chunk_type + payload
    crc = binascii.crc32(chunk_data) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_data + struct.pack(">I", crc)


def _parse_png(data):
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("输入文件不是有效的 PNG 文件")

    chunks = []
    offset = len(PNG_SIGNATURE)
    data_length = len(data)
    while offset < data_length:
        if data_length - offset < 12:
            raise ValueError("PNG chunk 头部不完整")

        payload_length = struct.unpack(">I", data[offset:offset + 4])[0]
        chunk_end = offset + 12 + payload_length
        if chunk_end > data_length:
            raise ValueError("PNG chunk 数据不完整")

        chunk_type_start = offset + 4
        chunk_type = data[chunk_type_start:chunk_type_start + 4]
        payload_start = chunk_type_start + 4
        payload_end = payload_start + payload_length
        payload = data[payload_start:payload_end]
        stored_crc = struct.unpack(">I", data[payload_end:chunk_end])[0]
        calculated_crc = binascii.crc32(chunk_type + payload) & 0xFFFFFFFF
        if stored_crc != calculated_crc:
            raise ValueError(f"PNG chunk {chunk_type.decode('ascii', 'replace')} CRC 校验失败")

        chunks.append((chunk_type, payload, data[offset:chunk_end]))
        offset = chunk_end
        if chunk_type == IEND_CHUNK:
            return chunks, offset

    raise ValueError("PNG 文件缺少 IEND chunk")


def _get_text_key(payload):
    separator = payload.find(b"\x00")
    if separator <= 0:
        return None
    return payload[:separator]


def _replace_text_chunks(data, key, text):
    chunks, iend_end = _parse_png(data)
    # iTXt: keyword, compression flag/method, language, translated keyword, UTF-8 text.
    new_chunk = _build_chunk(ITEXT_CHUNK, key + b"\x00\x00\x00\x00\x00" + text)
    output_chunks = [PNG_SIGNATURE]
    inserted = False

    for chunk_type, payload, raw_chunk in chunks:
        if chunk_type in TEXT_CHUNKS:
            continue
        if chunk_type == IEND_CHUNK and not inserted:
            output_chunks.append(new_chunk)
            inserted = True
        output_chunks.append(raw_chunk)

    if not inserted:
        raise ValueError("PNG 文件缺少 IEND chunk")
    return b"".join(output_chunks) + data[iend_end:]


def _tensor_to_png(image, compress_level=4):
    if image.ndim != 3:
        raise ValueError("图片必须是 H x W x C 格式")
    if image.shape[2] not in (1, 3, 4):
        raise ValueError("图片通道数必须是1、3或4")

    pixels = image.detach().to(device="cpu", dtype=torch.float32)
    pixels = pixels.clamp(0.0, 1.0).mul(255.0).round().to(dtype=torch.uint8).numpy()
    if pixels.shape[2] == 1:
        pixels = pixels[:, :, 0]

    with io.BytesIO() as buffer:
        Image.fromarray(np.ascontiguousarray(pixels)).save(buffer, format="PNG", compress_level=compress_level)
        return buffer.getvalue()


def _attach_png_frames(image, frames):
    output_image = image.clone()
    try:
        setattr(output_image, PNG_FRAMES_ATTRIBUTE, tuple(frames))
    except Exception as exc:
        raise RuntimeError("当前 PyTorch 版本不支持在 IMAGE tensor 上传递 PNG 数据") from exc
    return output_image


def _load_image_tensor(image_path):
    img = Image.open(image_path)

    output_images = []
    width = None
    height = None
    for frame in ImageSequence.Iterator(img):
        frame = ImageOps.exif_transpose(frame)
        if frame.mode == "I":
            frame = frame.point(lambda i: i * (1 / 255))
        image = frame.convert("RGB")

        if not output_images:
            width, height = image.size
        if image.size[0] != width or image.size[1] != height:
            continue

        image = np.array(image).astype(np.float32) / 255.0
        output_images.append(torch.from_numpy(image)[None,])

        if img.format == "MPO":
            break

    if not output_images:
        raise ValueError(f"无法加载图片: {image_path}")
    if len(output_images) > 1:
        return torch.cat(output_images, dim=0)
    return output_images[0]


def _read_text_from_png(data, key):
    chunks, _ = _parse_png(data)
    result = ""
    for chunk_type, payload, _ in chunks:
        if chunk_type == TEXT_CHUNK and _get_text_key(payload) == key:
            separator = payload.find(b"\x00")
            result = payload[separator + 1:].decode("latin-1")
        elif chunk_type == ITEXT_CHUNK and _get_text_key(payload) == key:
            separator = payload.find(b"\x00")
            remainder = payload[separator + 1:]
            if len(remainder) < 2:
                continue

            compression_flag = remainder[0]
            compression_method = remainder[1]
            remainder = remainder[2:]
            language_separator = remainder.find(b"\x00")
            if language_separator < 0:
                continue
            remainder = remainder[language_separator + 1:]
            translated_separator = remainder.find(b"\x00")
            if translated_separator < 0:
                continue

            text = remainder[translated_separator + 1:]
            if compression_flag == 1 and compression_method == 0:
                text = zlib.decompress(text)
            elif compression_flag != 0:
                continue
            result = text.decode("utf-8")
    return result


class WritePngText:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片": ("IMAGE",),
                "文本键": ("STRING", {"default": "Comment", "multiline": False}),
                "字符串": ("STRING", {"default": "", "multiline": True}),
                "filename_prefix": ("STRING", {"default": "ComfyUI_iTXt", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图片", "保存路径")
    FUNCTION = "write_text"
    OUTPUT_NODE = True
    CATEGORY = "百宝箱/图像"

    def write_text(self, 图片, 文本键, 字符串, filename_prefix="ComfyUI_iTXt"):
        key = _encode_text_key(文本键)
        try:
            text = str(字符串).encode("utf-8", "strict")
        except UnicodeEncodeError as exc:
            raise ValueError("字符串无法编码为 UTF-8") from exc

        if not isinstance(图片, torch.Tensor) or 图片.ndim != 4:
            raise ValueError("图片必须是 ComfyUI IMAGE 类型")

        png_frames = []
        for image in 图片:
            png_data = _tensor_to_png(image, self.compress_level)
            png_frames.append(_replace_text_chunks(png_data, key, text))

        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            self.output_dir,
            图片[0].shape[1],
            图片[0].shape[0],
        )

        results = []
        saved_paths = []
        for batch_number, png_data in enumerate(png_frames):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            full_path = os.path.join(full_output_folder, file)
            with open(full_path, "wb") as output_file:
                output_file.write(png_data)

            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            saved_paths.append(full_path)
            counter += 1

        output_image = _attach_png_frames(图片, png_frames)
        return {"ui": {"images": results}, "result": (output_image, "\n".join(saved_paths))}


class ReadPngText:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "文本键": ("STRING", {"default": "Comment", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图片", "字符串")
    FUNCTION = "read_text"
    CATEGORY = "百宝箱/图像"

    def read_text(self, image, 文本键):
        key = _encode_text_key(文本键)
        image_path = folder_paths.get_annotated_filepath(image)
        with open(image_path, "rb") as input_file:
            result = _read_text_from_png(input_file.read(), key)
        return (_load_image_tensor(image_path), result)

    @classmethod
    def IS_CHANGED(cls, image, 文本键):
        image_path = folder_paths.get_annotated_filepath(image)
        hasher = hashlib.sha256()
        with open(image_path, "rb") as input_file:
            hasher.update(input_file.read())
        hasher.update(str(文本键).encode("utf-8", "strict"))
        return hasher.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, image, 文本键):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True
