FROM nvcr.io/nvidia/pytorch:21.09-py3

RUN apt update && \
    apt install -y zip htop screen libgl1-mesa-glx

RUN pip install seaborn thop

COPY chess_moves_classification_model/requirements.txt /tmp/
RUN pip install --no-cache --requirement /tmp/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113/
RUN pip install --no-cache opencv-python==4.5.5.64 --extra-index-url https://download.pytorch.org/whl/cu113/
