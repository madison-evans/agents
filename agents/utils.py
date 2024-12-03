
import json
import logging


class CustomFormatter(logging.Formatter):
        GREEN = "\033[92m"
        RESET = "\033[0m"

        def format(self, record):
            if "Tool Call" in record.msg:
                record.msg = f"{self.GREEN}{record.msg}{self.RESET}"
            if isinstance(record.args, dict):
                record.msg += "\n" + json.dumps(record.args, indent=4)
                record.args = ()  
            return super().format(record)