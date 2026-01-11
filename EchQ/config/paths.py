from pathlib import Path

class Paths:
    """项目路径类
    
    包含项目中常用路径
    """

    ROOT = Path(__file__).parent.parent.parent
    
    # 数据目录
    DATA = ROOT / "data"
    VECTOR_DB = DATA / "vector_db"
    CHROMA_DB = VECTOR_DB / "chroma"
    
    # 资源目录
    RESOURCES = ROOT / "resources"
    SOUNDS = RESOURCES / "sounds"

    # 源代码目录
    SRC = ROOT / "EchQ"
    CONFIG = SRC / "config"
    PROMPT_FILE = CONFIG / "prompt.txt"


__all__ = ["Paths"]
