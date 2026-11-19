# Public benchmark assets only; do not mount the user's home, keys or credentials.
FROM eclipse-temurin:21-jdk@sha256:0a3f329821dcadf54b2a3033a15e2e9f007069a9db43d7f51897e6b80e726b07 AS java
FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df
COPY --from=java /opt/java/openjdk /opt/java/openjdk
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="/opt/java/openjdk/bin:${PATH}"
RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/upstream-requirements.txt
# Install the recorded CPU wheel before requirements so pip does not pull a
# multi-gigabyte CUDA build; requirements then sees torch as satisfied.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.9.1
RUN pip install --no-cache-dir -r /tmp/upstream-requirements.txt openai==1.109.1
RUN python -c "import torch, openai; assert torch.__version__.startswith('2.9.1'), torch.__version__; assert openai.__version__ == '1.109.1', openai.__version__"
RUN useradd -m -u 1000 benchmark && mkdir -p /bench && chown benchmark:benchmark /bench \
    && touch /SHOPPINGBENCH_ISOLATED
USER benchmark
ENV HF_HOME=/tmp/huggingface
ENV SHOPPINGBENCH_ISOLATED=1
WORKDIR /bench
