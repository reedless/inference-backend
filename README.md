# inference-backend, instructions for running on tablet

Install Termux: https://github.com/termux/termux-app#Installation

Install Ubuntu in Termux: https://github.com/MFDGaming/ubuntu-in-termux

`cd ubuntu-in-termux`

`./startubuntu.sh`

`apt-get update && apt-get install -y libopencv-dev python3-pip git` and possibly `openssh-client`

Clone this repository (might need a new SSH key)

`apt install python3.10-venv`

`cd inference-backend`

`bash startup.sh`

Go to `localhost:5000`
