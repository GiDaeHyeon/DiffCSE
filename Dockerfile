FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.12-cuda11.6.1

RUN pip3 install transformers pyyaml pytorch_lightning