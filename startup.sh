python3 -m venv ~/.env && \
source ~/.env/bin/activate && \
pip3 install -r ~/inference-backend/requirements.txt && \
pip3 install torch==1.10.2 && \
python3 ~/inference-backend/inference_backend.py