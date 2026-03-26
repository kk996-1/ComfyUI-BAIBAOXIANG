import time
from datetime import datetime

class TimeString:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "前缀": ("STRING", {"default": "", "multiline": False}),
                "后缀": ("STRING", {"default": "", "multiline": False}),
                "年": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭"}),
                "月": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭"}),
                "日": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭"}),
                "时": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭"}),
                "分": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭"}),
                "秒": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("字符串",)
    FUNCTION = "get_time_string"
    CATEGORY = "百宝箱/工具"
    
    # 设置为 output_node = True 可以让节点在没有连接输出时也能执行（可选，但对于这种生成器节点通常建议）
    # 但如果是纯粹的数据处理节点，通常不需要。考虑到这只是生成字符串，保持默认即可。
    
    # 每次运行时强制更新，因为时间是变化的
    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_time_string(self, 前缀, 后缀, 年, 月, 日, 时, 分, 秒):
        """
        输出时间字符串节点
        """
        now = datetime.now()
        parts = []
        
        if 前缀:
            parts.append(前缀)
            
        if 年:
            parts.append(now.strftime("%Y"))
        if 月:
            parts.append(now.strftime("%m"))
        if 日:
            parts.append(now.strftime("%d"))
        if 时:
            parts.append(now.strftime("%H"))
        if 分:
            parts.append(now.strftime("%M"))
        if 秒:
            parts.append(now.strftime("%S"))
            
        if 后缀:
            parts.append(后缀)
            
        result = "_".join(parts)
        
        return (result,)
