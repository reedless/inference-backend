python3 -m venv .env && \
source .env/bin/activate && \
pip3 install -r requirements.txt && \
pip3 install torch==1.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
python3 inference_backend.py