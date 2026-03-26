from .nodes.image_stitching import ImageStitching
from .nodes.time_string import TimeString
from .nodes.text_to_list import TextToList
from .nodes.auto_batch_submit import AutoBatchSubmit
from .nodes.keyword_image_batch import KeywordImageBatch

NODE_CLASS_MAPPINGS = {
    "KK_ImageStitching": ImageStitching,
    "KK_TimeString": TimeString,
    "KK_TextToList": TextToList,
    "KK_AutoBatchSubmit": AutoBatchSubmit,
    "KK_KeywordImageBatch": KeywordImageBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KK_ImageStitching": "图像自动拼接",
    "KK_TimeString": "输出时间字符串",
    "KK_TextToList": "文本转字符串列表",
    "KK_AutoBatchSubmit": "自动批量提交任务",
    "KK_KeywordImageBatch": "根据关键词输出对应图片-自动批量提交任务"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
