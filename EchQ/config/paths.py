from pathlib import Path

class Paths:
    """项目路径类
    
    包含项目中常用路径
    """

    ROOT = Path(__file__).parent.parent.parent
    
    DATA = ROOT / "data"
    
    RESOURCES = ROOT / "resources"
    SOUNDS = RESOURCES / "sounds"

    SRC = ROOT / "EchQ"
    CONFIG = SRC / "config"
    PROMPT_FILE = CONFIG / "prompt.txt"


__all__ = ["Paths"]
