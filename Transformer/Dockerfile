FROM registry.cn-shanghai.aliyuncs.com/tcc_public/python:3.10
ADD . /app
WORKDIR /app
Run /usr/local/bin/python -m pip install --upgrade pip
RUN pip install torch==2.0.1+cpu \
     --index-url https://download.pytorch.org/whl/cpu
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /app/requirements.txt
CMD ["sh", "/app/run.sh"]