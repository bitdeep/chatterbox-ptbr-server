FROM docker.io/library/python:3.11-slim-bookworm@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/models/hf \
    CHATTERBOX_MODELS_DIR=/models \
    CHATTERBOX_REFERENCE_DIR=/data/reference_audio

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir \
      chatterbox-tts==0.1.7 \
      setuptools==80.9.0 \
      fastapi==0.141.1 \
      uvicorn==0.52.4 \
      python-multipart==0.0.32 \
      soundfile==0.14.0

RUN pip install --no-cache-dir --no-deps --no-build-isolation --force-reinstall \
      "chatterbox-tts @ https://github.com/resemble-ai/chatterbox/archive/5de7a54aa4e5e2baadb0182dde554908b48b85c2.tar.gz#sha256=003f8c85dcfeb2d91b3a6f97f43b74703d15131e987dfabb7f3d9aee7c0da2cf" \
    && python -c "import inspect; from chatterbox.mtl_tts import ChatterboxMultilingualTTS as M; assert 't3_model' in inspect.signature(M.from_local).parameters, 'chatterbox sem suporte a V3'"

COPY server.py /app/server.py
RUN mkdir -p /models /data/reference_audio \
    # `/tts` tem de ler o pedido do CORPO (o 422 de 08/09 era o FastAPI pedindo `req` na query).
    && python -c "import server; app = server.criar_app(); r = [x for x in app.routes if getattr(x, 'path', '') == '/tts'][0]; assert r.body_field is not None, '/tts sem corpo'"

EXPOSE 8004
CMD ["python", "server.py"]
