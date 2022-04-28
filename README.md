# inference-backend, instructions for running on tablet

Install Termux: https://github.com/termux/termux-app#Installation

Install Ubuntu in Termux: https://github.com/MFDGaming/ubuntu-in-termux

`cd ubuntu-in-termux`

`./startubuntu.sh`

`apt-get update && apt-get install -y libopencv-dev python3-pip git` and possibly `openssh-client`

Clone this repository (might need a new SSH key)

`apt install python3.10-venv`

`cd inference-backend`

`python3 -m venv venv`

`source venv/bin/activate`

`pip3 install -r requirements.txt`

`pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu`

`python3 inference_backend.py`

Go to `localhost:5000`
