# Upstream components

- Server integration: MIT, see [LICENSE](LICENSE).
- [Chatterbox](https://github.com/resemble-ai/chatterbox): upstream model implementation. The Dockerfile fixes revision `5de7a54aa4e5e2baadb0182dde554908b48b85c2` and verifies the source archive SHA-256.
- Checkpoints: `ResembleAI/chatterbox` and `ResembleAI/Chatterbox-Multilingual-pt-br` on Hugging Face. Weights are downloaded separately, never embedded in this repository. Preserve their model cards and licenses when redistributing weights.
- Chatterbox's PerTh watermark remains part of the model's synthesis path.
- FastAPI, Pydantic, PyTorch, soundfile and the container base retain their respective upstream licenses.

The API keeps a compatible subset of the reference-upload and synthesis routes used by Chatterbox-TTS-Server integrations. It does not vendor that server.
