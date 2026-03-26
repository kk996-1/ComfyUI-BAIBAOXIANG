import torch
import torch.nn.functional as F

class ImageStitching:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "拼接方向": (["竖向", "横向"], {"default": "竖向"}),
                "统一图像尺寸": ("BOOLEAN", {"default": False, "label_on": "开启", "label_off": "关闭"}),
                "间距": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "间距颜色": (["黑色", "白色"], {"default": "黑色"}),
                "数量限制": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stitch_images"
    CATEGORY = "百宝箱/图像"

    def stitch_images(self, images, 拼接方向, 统一图像尺寸, 间距, 间距颜色, 数量限制):
        """
        图像自动拼接节点
        :param images: 输入的图片批次 (Batch, H, W, C) 或 图片列表
        :param 拼接方向: 拼接方向 ("竖向" 或 "横向")
        :param 统一图像尺寸: 是否统一图像尺寸
        :param 间距: 图片间距 (像素)
        :param 间距颜色: 间距颜色 ("黑色" 或 "白色")
        :param 数量限制: 换行/换列的数量限制
        :return: 拼接后的大图
        """
        
        # 1. 统一输入格式为 List[Tensor(H, W, C)]
        img_list = []
        if isinstance(images, torch.Tensor):
            # images shape: [B, H, W, C]
            for i in range(images.shape[0]):
                img_list.append(images[i])
        elif isinstance(images, list):
            img_list = images
        else:
            # 如果不是 Tensor 也不是 List，直接返回原输入（作为保底）
            return (images,)

        if not img_list:
            # 返回一个空的 64x64 黑色图片防止报错
            return (torch.zeros((1, 64, 64, 3)),)

        # 获取设备和数据类型，确保创建的 tensor 与输入一致
        device = img_list[0].device
        dtype = img_list[0].dtype

        # 2. 获取所有图片的最大尺寸
        # img.shape is (H, W, C)
        # 注意：如果输入是 Tensor Batch，所有图片尺寸是一样的。
        # 只有当输入是 List[Tensor] 且确实有不同尺寸时，这里的 max 才有意义。
        max_h = max(img.shape[0] for img in img_list)
        max_w = max(img.shape[1] for img in img_list)

        # 3. 统一尺寸逻辑
        processed_imgs = []
        for img in img_list:
            h, w = img.shape[:2]
            
            if 统一图像尺寸:
                target_h, target_w = h, w
                if 拼接方向 == "竖向":
                    # 竖向模式：统一宽度，高度等比缩放
                    # 只有当宽度不一致时才需要调整
                    if w != max_w:
                        scale = max_w / w
                        target_w = max_w
                        target_h = int(h * scale)
                else:
                    # 横向模式：统一高度，宽度等比缩放
                    # 只有当高度不一致时才需要调整
                    if h != max_h:
                        scale = max_h / h
                        target_h = max_h
                        target_w = int(w * scale)
                
                # 如果计算出的目标尺寸与原尺寸不同，则进行 resize
                if target_h != h or target_w != w:
                    # F.interpolate 需要输入 (N, C, H, W)
                    # img 是 (H, W, C) -> permute -> (C, H, W) -> unsqueeze -> (1, C, H, W)
                    permuted = img.permute(2, 0, 1).unsqueeze(0)
                    
                    # 使用 bilinear 插值
                    resized = F.interpolate(permuted, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    
                    # 转回 (H, W, C)
                    # resized 是 (1, C, H', W') -> squeeze -> (C, H', W') -> permute -> (H', W', C)
                    processed_img = resized.squeeze(0).permute(1, 2, 0)
                    processed_imgs.append(processed_img)
                else:
                    processed_imgs.append(img)
            else:
                processed_imgs.append(img)
        
        # 4. 拼接逻辑
        
        # 将图片按 limit 分组
        groups = []
        for i in range(0, len(processed_imgs), 数量限制):
            groups.append(processed_imgs[i:i+数量限制])
            
        # 确定填充颜色值
        fill_value = 0.0 if 间距颜色 == "黑色" else 1.0
        
        if 拼接方向 == "竖向":
            # 竖向模式：每 limit 张图组成一列 (Column)
            # groups 中的每个元素是一个列的图片列表
            
            # 计算每一列的最终宽高
            col_dims = [] # List of (width, height)
            for col_imgs in groups:
                # 列宽 = 该列图片中最大的宽度
                col_w = max(img.shape[1] for img in col_imgs)
                # 列高 = 该列所有图片高度之和 + 间距
                col_h = sum(img.shape[0] for img in col_imgs) + 间距 * (len(col_imgs) - 1)
                col_dims.append((col_w, col_h))
            
            # 计算画布总宽高
            # 总宽 = 所有列宽之和 + 列间距
            total_w = sum(d[0] for d in col_dims) + 间距 * (len(groups) - 1)
            # 总高 = 所有列中最高的那个高度
            total_h = max(d[1] for d in col_dims)
            
            # 创建画布
            canvas = torch.full((total_h, total_w, 3), fill_value, dtype=dtype, device=device)
            
            # 开始粘贴
            current_x = 0
            for i, col_imgs in enumerate(groups):
                col_w, col_h = col_dims[i]
                current_y = 0
                for img in col_imgs:
                    h, w, c = img.shape
                    # 粘贴图片 (默认左对齐)
                    # 边界检查防止溢出 (理论上不会，但在 float 计算时可能有一像素误差)
                    canvas[current_y:current_y+h, current_x:current_x+w, :] = img
                    current_y += h + 间距
                current_x += col_w + 间距
                
        else:
            # 横向模式：每 limit 张图组成一行 (Row)
            # groups 中的每个元素是一行的图片列表
            
            # 计算每一行的最终宽高
            row_dims = [] # List of (width, height)
            for row_imgs in groups:
                # 行高 = 该行图片中最大的高度
                row_h = max(img.shape[0] for img in row_imgs)
                # 行宽 = 该行所有图片宽度之和 + 间距
                row_w = sum(img.shape[1] for img in row_imgs) + 间距 * (len(row_imgs) - 1)
                row_dims.append((row_w, row_h))
                
            # 计算画布总宽高
            # 总宽 = 所有行中最宽的那个宽度
            total_w = max(d[0] for d in row_dims)
            # 总高 = 所有行高之和 + 行间距
            total_h = sum(d[1] for d in row_dims) + 间距 * (len(groups) - 1)
            
            # 创建画布
            canvas = torch.full((total_h, total_w, 3), fill_value, dtype=dtype, device=device)
            
            # 开始粘贴
            current_y = 0
            for i, row_imgs in enumerate(groups):
                row_w, row_h = row_dims[i]
                current_x = 0
                for img in row_imgs:
                    h, w, c = img.shape
                    # 粘贴图片 (默认顶对齐)
                    canvas[current_y:current_y+h, current_x:current_x+w, :] = img
                    current_x += w + 间距
                current_y += row_h + 间距

        # 返回结果
        # ComfyUI 期望输出格式为 (1, H, W, C) 的 Tensor (如果是单张图)
        return (canvas.unsqueeze(0),)
