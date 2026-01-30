uv init
uv venv --seed --python 3.13
uv pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
uv pip install -U "paddleocr[doc-parser]"