from datetime import datetime

class DatetimeUtils:
    """日期时间处理工具类"""

    def __new__(cls):
        # 禁止实例化
        raise TypeError("DatetimeUtils类不可被实例化")
    
    @staticmethod
    def format_relative_time(timestamp: int | float) -> str:
        """格式化时间戳为相对时间字符串

        注意: 仅支持处理过去的时间, 未来的时间将被视为“刚刚”

        Args:
            timestamp: 时间戳(秒级)
        """
        now = datetime.now()
        dt = datetime.fromtimestamp(timestamp)
        delta = now - dt
        
        seconds = delta.total_seconds()
        days = delta.days

        match (days, seconds):
            # 未来的时间视为"刚刚"
            case (d, _) if d < 0:
                return "刚刚"
            
            # 使用 round 进行四舍五入计算
            case (d, _) if d >= 320: # 约 10.5 个月以上
                years = round(d / 365)
                return f"{years} 年前" if years > 0 else "1 年前"
            
            case (d, _) if d >= 25: # 约 3.5 周以上
                months = round(d / 30)
                return f"{months} 个月前"
            
            case (d, _) if d >= 7:
                weeks = round(d / 7)
                return f"{weeks} 周前"
            
            case (1, _):
                return "昨天"
            
            case (2, _):
                return "前天"
            
            case (d, _) if d >= 3:
                return f"{d} 天前"
            
            case (_, s) if s >= 3600:
                hours = round(s / 3600)
                return f"{hours} 小时前"
            
            case (_, s) if s >= 600:
                return "最近"
            
            case _:
                return "刚刚"


__all__ = ["DatetimeUtils"]
