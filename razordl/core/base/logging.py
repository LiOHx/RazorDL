import logging as _logging
import os

# 暴露常用常量，方便直接作为 logging 替代品使用
from logging import INFO, DEBUG, WARNING, ERROR, CRITICAL, StreamHandler, FileHandler

class RankZeroLogger(_logging.Logger):
    """
    A custom Logger that only logs info messages on the main process (LOCAL_RANK=0).
    Other levels (warning, error, etc.) behave normally.
    """
    def info(self, msg, *args, **kwargs):
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            kwargs['stacklevel'] = kwargs.get('stacklevel', 1) + 1
            super().info(msg, *args, **kwargs)

# 自动设置为默认 Logger 类
_logging.setLoggerClass(RankZeroLogger)

# 暴露 getLogger，允许用户使用 'from base import logging' 后直接调用 logging.getLogger
def getLogger(name=None):
    return _logging.getLogger(name)

def basicConfig(**kwargs):
    """
    Wrapper for logging.basicConfig with project defaults.
    """
    # 设置默认参数，如果用户提供了参数则使用用户的
    kwargs.setdefault('level', INFO)
    kwargs.setdefault('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    kwargs.setdefault('handlers', [StreamHandler()])
    
    _logging.basicConfig(**kwargs)
