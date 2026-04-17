"""Hard-coded Qwen client settings for AirVLN V0.

Replace the placeholder values below before running against a real Qwen API or
local Qwen service. They are intentionally not read from environment variables
so the V0 pipeline uses one explicit configuration source.
"""

QWEN_API_KEY = "PASTE_YOUR_QWEN_API_KEY_HERE"
QWEN_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_TEXT_MODEL = "qwen-plus"
QWEN_VLM_MODEL = "qwen-vl-plus"

# Local command contracts:
# - LLM command receives JSON on stdin: {"system": "...", "user": "..."}
# - VLM command receives JSON on stdin:
#   {"prompt": "...", "actions": [...], "images_base64_png": [...]}
# Both commands must print a JSON object on stdout.
QWEN_LOCAL_LLM_COMMAND = "python tools/qwen_local_llm.py"
QWEN_LOCAL_VLM_COMMAND = "python tools/qwen_local_vlm.py"

