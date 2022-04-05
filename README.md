# inference-backend, instructions for running on tablet

Install Termux: https://github.com/termux/termux-app#Installation

Install Ubuntu in Termux: https://github.com/MFDGaming/ubuntu-in-termux

`apt-get update && apt-get install -y libopencv-dev python3-pip git` and possibly `openssh-client`

Clone this repository (might need a new SSH key)

`apt install python3.10-venv`

Create a new venv (takes a while longer than usual?)

`pip3 install -r requirements.txt`

`python3 inference_backend.py`