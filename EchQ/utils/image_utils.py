import base64
from io import BytesIO
import logging

from PIL import Image
import httpx

logger = logging.getLogger(__name__)


class ImageUtils:
    """图片处理工具类"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super(ImageUtils, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, timeout: float = 10.0):
        """初始化 ImageUtils 实例
        
        Args:
            timeout: HTTP 请求超时时间 (秒)
        """
        if self._initialized:
            return
        self.timeout = timeout
        self.client = None  # 初始不创建, 在异步方法中延迟创建
        self._initialized = True

    def _ensure_client(self):
        """延迟创建 AsyncClient, 确保其绑定到正确的 Event Loop"""
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)
            )
        return self.client

    async def close(self):
        """关闭全局连接池"""
        if self.client:
            await self.client.aclose()
            logger.info("ImageUtils HTTP 客户端已关闭")

    async def get_remote_image_b64(
        self,
        url: str,
        max_mb: int = 10,
        max_size: int = 1024,
        quality: int = 85
    ) -> str:
        """从 URL 下载并直接转为压缩后的 Base64
        
        集成了流式下载, 大小检查, 格式转换和压缩功能. 格式为 JPEG / GIF.

        Args:
            url: 图片的远程 URL 地址
            max_mb: 图片的最大允许体积 (MB)
            max_size: 图片的最大边长, 超过则进行缩放
            quality: 压缩质量 (1-100)

        Returns:
            Base64 编码的图片字符串, 失败则返回空字符串
        """
        client = self._ensure_client()

        max_bytes = max_mb * 1024 * 1024
        
        try:
            async with client.stream("GET", url, follow_redirects=True) as response:
                # 检查响应状态
                if response.status_code != 200:
                    logger.error(f"无法访问图片 URL, 状态码: {response.status_code}")
                    return ""

                # 检查 Header 中的大小 (如果提供)
                content_length = int(response.headers.get("Content-Length", 0))
                if content_length > max_bytes:
                    logger.warning(f"图片体积({content_length} bytes)超过限制: {max_mb}MB")
                    return ""

                # 流式读取并累加
                image_data = bytearray()
                async for chunk in response.aiter_bytes():
                    image_data.extend(chunk)
                    if len(image_data) > max_bytes:
                        logger.error(f"图片下载体积过大, 中断连接")
                        return ""

            # 压缩转码
            return ImageUtils.bytes_to_base64(bytes(image_data), max_size=max_size, quality=quality)

        except Exception as e:
            logger.error(f"处理远程图片流程出错: {e}")
            return ""
    
    @staticmethod
    def bytes_to_base64(image_bytes: bytes, max_size: int = 1024, quality: int = 85) -> str:
        """将原始字节流压缩并转换为 Base64 字符串.  格式为 JPEG / GIF.
        
        Args:
            image_bytes: 原始图片的字节流
            max_size: 图片的最大边长, 超过则进行缩放
            quality: 压缩质量 (1-100)

        Returns:
            Base64 编码的图片字符串, 失败则返回空字符串
        """
        try:
            # 使用 BytesIO 包装字节流, 避免直接操作磁盘
            with Image.open(BytesIO(image_bytes)) as img:
                original_format = img.format
                
                # 统一转换为 RGB (处理 RGBA 或透明 P 模式，防止转 JPEG 报错)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # 保持宽高比进行缩小
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # 写入内存缓冲区
                output_buffer = BytesIO()
                
                if original_format == "GIF":
                    # GIF 保持原格式以支持动画
                    img.save(output_buffer, format="GIF", optimize=True)
                    prefix = "data:image/gif;base64,"
                else:
                    # 静态图片统一导出为 JPEG, 因为 JPEG 在同等清晰度下 Base64 长度最短
                    img.save(output_buffer, format="JPEG", quality=quality, optimize=True)
                    prefix = "data:image/jpeg;base64,"
                
                # 编码
                encoded_str = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
                
                logger.info(f"图片处理成功: 原始格式 {original_format}, 缩放至 {img.size}")
                return prefix + encoded_str
                
        except Exception as e:
            logger.error(f"图片处理失败: {e}")
            return ""


# 创建单例实例
image_utils = ImageUtils()

__all__ = ["image_utils"]
