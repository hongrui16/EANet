import logging
import os
import sys

class colorlogger:
    def __init__(self, log_dir, log_name='train_logs.txt'):
        self._logger = logging.getLogger(log_name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        if self._logger.handlers:
            self._logger.handlers.clear()

        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, log_name)

        # 文件日志（无颜色）
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%m-%d %H:%M:%S"
        ))
        self._logger.addHandler(file_handler)

        # 控制台日志（带颜色）
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self._get_color_formatter())
        self._logger.addHandler(console_handler)

    def _get_color_formatter(self):
        # ANSI 颜色
        COLORS = {
            'DEBUG': '\033[94m',   # 蓝色
            'INFO': '\033[92m',    # 绿色
            'WARNING': '\033[93m', # 黄色
            'ERROR': '\033[91m',   # 红色
            'CRITICAL': '\033[1;91m' # 加粗红
        }
        RESET = '\033[0m'

        class ColorFormatter(logging.Formatter):
            def format(self, record):
                level_color = COLORS.get(record.levelname, '')
                message = super().format(record)
                return f"{level_color}{message}{RESET}"

        return ColorFormatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%m-%d %H:%M:%S"
        )

    # 日志方法
    def debug(self, msg):    self._logger.debug(str(msg))
    def info(self, msg):     self._logger.info(str(msg))
    def warning(self, msg):  self._logger.warning(str(msg))
    def error(self, msg):    self._logger.error(str(msg))
    def critical(self, msg): self._logger.critical(str(msg))


if __name__ == "__main__":
    logger = colorlogger(log_dir='logs')
    logger.info("This is an info message.")
    logger.error("This is an error message.")
    
    