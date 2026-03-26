import json
import ast

class TextToList:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "文本": ("STRING", {"default": "[\n\"text1\",\n\"text2\"\n]", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("字符串列表",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "parse_text"
    CATEGORY = "百宝箱/文本"

    def parse_text(self, 文本):
        """
        将文本解析为字符串列表
        """
        try:
            # 1. 尝试作为 JSON 解析
            data = json.loads(文本)
        except json.JSONDecodeError:
            try:
                # 2. 尝试作为 Python 字面量解析 (支持单引号等)
                data = ast.literal_eval(文本)
            except (ValueError, SyntaxError):
                data = [文本]

        # 确保结果是列表
        if not isinstance(data, list):
            # 如果解析出来是字典或其他类型，包裹成列表
            data = [data]
        
        # 确保列表中的每个元素都是字符串
        # 如果元素是 None 或其他类型，转为字符串
        result = []
        for item in data:
            if isinstance(item, (dict, list)):
                # 如果元素本身是复杂结构，转为 JSON 字符串
                result.append(json.dumps(item, ensure_ascii=False))
            else:
                result.append(str(item))
        
        return (result,)
