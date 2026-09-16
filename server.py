"""Chatterbox Multilingual V3 server with a Brazilian Portuguese checkpoint.

Model loading and reference audio stay on the inference node. No customer data
is bundled with the server. See README.md for the HTTP contract and operations.
"""
import argparse
import gc
import io
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

from pydantic import BaseModel, Field

MODELS_DIR = Path(os.environ.get("CHATTERBOX_MODELS_DIR", "/models"))
REFERENCE_DIR = Path(os.environ.get("CHATTERBOX_REFERENCE_DIR", "/data/reference_audio"))
BASE_REPO = os.environ.get("CHATTERBOX_BASE_REPO", "ResembleAI/chatterbox")
T3_REPO = os.environ.get("CHATTERBOX_T3_REPO", "ResembleAI/Chatterbox-Multilingual-pt-br")
T3_FILE = os.environ.get("CHATTERBOX_T3_FILE", "t3_pt_br.safetensors")
S3GEN_REPO = os.environ.get("CHATTERBOX_S3GEN_REPO", T3_REPO)
S3GEN_FILE = os.environ.get("CHATTERBOX_S3GEN_FILE", "s3gen_v3.pt")
DEVICE = os.environ.get("CHATTERBOX_DEVICE", "cuda")
PORT = int(os.environ.get("PORT", "8004"))
MAX_CHUNK = 500
MAX_REFERENCE_BYTES = 8 * 1024 * 1024
SILENCE_BETWEEN_CHUNKS_S = 0.2
# Pausa PEDIDA pelo chamador, na mesma marcação do Kokoro (`[pause:2.5s]`): um formato só para a
# frota inteira. Sem isto, quem quer silêncio entre frases precisa cortar o texto e emendar o
# áudio por fora — trabalho a mais e uma junta a mais para cada pausa.
PAUSE_TAG = re.compile(r"\[pause:(\d+(?:\.\d+)?)s\]", re.IGNORECASE)
# ⛔ Sem teto de pausa: quem decide quanto silêncio o áudio tem é quem escreve o roteiro
# (dono, 16/09/2026). O limite que sobra é o tamanho do texto que o chamador manda.


def log(msg: str) -> None:
    print(f"[chatterbox] {msg}", flush=True)


def montar_checkpoint() -> Path:
    """Diretório que `from_local` lê: ve.pt + conds.pt (base), T3 + grapheme + s3gen (pack)."""
    from huggingface_hub import hf_hub_download

    ckpt = MODELS_DIR / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    plano = [
        (BASE_REPO, "ve.pt", "ve.pt"),
        (BASE_REPO, "conds.pt", "conds.pt"),
        (T3_REPO, "grapheme_mtl_merged_expanded_v1.json", "grapheme_mtl_merged_expanded_v1.json"),
        (T3_REPO, T3_FILE, T3_FILE),
        # `from_local` lê o nome fixo `s3gen.pt`; o pack V3 traz `s3gen_v3.pt`.
        (S3GEN_REPO, S3GEN_FILE, "s3gen.pt"),
    ]
    for repo, arquivo, destino in plano:
        alvo = ckpt / destino
        if alvo.is_symlink() and not alvo.exists():
            alvo.unlink()  # link pendurado de uma tentativa anterior
        if alvo.exists():
            continue
        log(f"baixando {repo}/{arquivo}")
        baixado = hf_hub_download(repo_id=repo, filename=arquivo, cache_dir=str(MODELS_DIR / "hf"))
        # O cache do HF devolve um SYMLINK relativo (`../../blobs/<hash>`); o checkpoint quer nomes
        # fixos. Liga-se ao BLOB real — link duro do symlink copiaria o alvo relativo e ficaria
        # pendurado dentro de `ckpt/` (foi o que aconteceu em 08/09/2026).
        real = os.path.realpath(baixado)
        try:
            os.link(real, alvo)
        except OSError:
            shutil.copyfile(real, alvo)
    return ckpt


def montar_modelo(ckpt: Path, device: str):
    """O que `ChatterboxMultilingualTTS.from_local` faz, tolerando o S3Gen V3 sem os buffers do tokenizer."""
    import torch
    from chatterbox.models.s3gen import S3Gen
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.tokenizers import MTLTokenizer
    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals
    from safetensors.torch import load_file as load_safetensors

    map_location = torch.device("cpu") if device in ("cpu", "mps") else None

    ve = VoiceEncoder()
    ve.load_state_dict(torch.load(ckpt / "ve.pt", map_location=map_location, weights_only=True))
    ve.to(device).eval()

    t3 = T3(T3Config.multilingual())
    t3_state = load_safetensors(ckpt / T3_FILE)
    if "model" in t3_state:
        t3_state = t3_state["model"][0]
    t3.load_state_dict(t3_state)
    t3.to(device).eval()

    s3gen = S3Gen()
    faltando, sobrando = s3gen.load_state_dict(
        torch.load(ckpt / "s3gen.pt", map_location=map_location, weights_only=True), strict=False
    )
    toleradas = set(getattr(s3gen, "ignore_state_dict_missing", ()))
    inesperadas = [k for k in faltando if k not in toleradas]
    if inesperadas or sobrando:
        raise RuntimeError(f"s3gen.pt não bate com a arquitetura: faltam {inesperadas[:5]}, sobram {list(sobrando)[:5]}")
    s3gen.to(device).eval()

    tokenizer = MTLTokenizer(str(ckpt / "grapheme_mtl_merged_expanded_v1.json"))

    conds = None
    if (voz_padrao := ckpt / "conds.pt").exists():
        conds = Conditionals.load(voz_padrao, map_location=map_location).to(device)
    return ChatterboxMultilingualTTS(t3, s3gen, ve, tokenizer, device, conds=conds)


# ---- modelo ----
_model = None
_model_lock = threading.Lock()
_gpu_lock = threading.Lock()
_load_error: str | None = None
_loaded_at: float | None = None


def carregar() -> None:
    global _model, _load_error, _loaded_at
    try:
        ckpt = montar_checkpoint()
        log(f"carregando T3={T3_FILE} S3Gen={S3GEN_FILE} em {DEVICE}")
        modelo = montar_modelo(ckpt, DEVICE)
        with _model_lock:
            _model = modelo
            _loaded_at = time.time()
        log(f"pronto: sr={modelo.sr}")
    except Exception as exc:  # noqa: BLE001 — a causa vai para o log e para /api/model-info
        _load_error = f"{type(exc).__name__}: {exc}"
        log(f"FALHA ao carregar: {_load_error}")


def descarregar() -> None:
    global _model, _loaded_at
    with _model_lock:
        _model = None
        _loaded_at = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def separar_pausas(texto: str) -> list[object]:
    """
    Texto em blocos: `str` é fala, `float` é silêncio em segundos.

    Pausa no COMEÇO e no FIM valem: silêncio antes de a voz entrar é recurso de indução, e é o que
    a pessoa escreve quando abre o roteiro com a marca. Pausas coladas somam, sem teto. Texto só
    de marcação devolve lista vazia — não há o que dizer.
    """
    blocos: list[object] = []
    for i, pedaco in enumerate(PAUSE_TAG.split(texto)):
        if i % 2 == 1:
            segundos = float(pedaco)
            if blocos and isinstance(blocos[-1], float):
                blocos[-1] += segundos
            else:
                blocos.append(segundos)
            continue
        fala = pedaco.strip()
        if fala:
            blocos.append(fala)
    if not any(isinstance(bloco, str) for bloco in blocos):
        return []
    return blocos


def dividir(texto: str, maximo: int) -> list[str]:
    limpo = re.sub(r"\s+", " ", texto).strip()
    if not limpo:
        return []
    if len(limpo) <= maximo:
        return [limpo]
    frases = re.findall(r"[^.!?…]+[.!?…]*\s*", limpo) or [limpo]
    trechos: list[str] = []
    atual = ""
    for frase in frases:
        f = frase.strip()
        if not f:
            continue
        if len(f) > maximo:
            if atual:
                trechos.append(atual.strip())
                atual = ""
            pedaco = ""
            for palavra in f.split(" "):
                if len((pedaco + " " + palavra).strip()) > maximo and pedaco:
                    trechos.append(pedaco.strip())
                    pedaco = ""
                pedaco = f"{pedaco} {palavra}"
            if pedaco.strip():
                atual = pedaco.strip()
            continue
        if len((atual + " " + f).strip()) > maximo:
            trechos.append(atual.strip())
            atual = f
        else:
            atual = f"{atual} {f}"
    if atual.strip():
        trechos.append(atual.strip())
    return trechos


def sintetizar(texto: str, referencia: Path, language: str, exaggeration: float, cfg_weight: float,
               temperature: float, seed: int | None, split_text: bool, chunk_size: int) -> bytes:
    import numpy as np
    import soundfile as sf
    import torch

    with _model_lock:
        modelo = _model
    if modelo is None:
        raise RuntimeError("modelo não carregado")
    # A pausa pedida corta ANTES de tudo: cada lado dela é sintetizado por conta e o silêncio
    # entra com a duração exata, no lugar da junta curta de sempre.
    blocos = separar_pausas(texto)
    partes: list[np.ndarray] = []
    silencio = np.zeros(int(modelo.sr * SILENCE_BETWEEN_CHUNKS_S), dtype=np.float32)
    indice = 0
    # Começa como "acabou de haver silêncio": antes da primeira fala não existe junta a preencher.
    ultimo_foi_pausa = True
    with _gpu_lock:
        for bloco in blocos:
            if isinstance(bloco, float):
                partes.append(np.zeros(int(modelo.sr * bloco), dtype=np.float32))
                ultimo_foi_pausa = True
                continue
            trechos = dividir(bloco, min(max(chunk_size, 50), MAX_CHUNK)) if split_text else [bloco.strip()]
            for trecho in trechos:
                if not trecho:
                    continue
                if seed is not None:
                    torch.manual_seed(seed + indice)
                indice += 1
                wav = modelo.generate(
                    trecho,
                    language_id=language,
                    audio_prompt_path=str(referencia),
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                )
                audio = wav.squeeze().detach().cpu().numpy().astype(np.float32)
                # Junta curta só entre trechos de FALA; depois de uma pausa pedida, ela sobraria.
                if not ultimo_foi_pausa:
                    partes.append(silencio)
                partes.append(audio)
                ultimo_foi_pausa = False
    saida = np.concatenate(partes) if partes else np.zeros(1, dtype=np.float32)
    pico = float(np.max(np.abs(saida))) if saida.size else 0.0
    if pico > 0.95:
        saida = saida * (0.95 / pico)
    buf = io.BytesIO()
    sf.write(buf, saida, modelo.sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50_000)
    voice_mode: str = "clone"
    reference_audio_filename: str | None = None
    predefined_voice_id: str | None = None
    language: str = "pt"
    exaggeration: float = Field(default=0.4, ge=0.25, le=2.0)
    cfg_weight: float = Field(default=0.5, ge=0.2, le=1.0)
    temperature: float = Field(default=0.7, ge=0.0, le=1.5)
    seed: int | None = 7
    split_text: bool = True
    chunk_size: int = Field(default=200, ge=50, le=MAX_CHUNK)
    output_format: str = "wav"


def criar_app():
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import JSONResponse, Response

    app = FastAPI(title="chatterbox-ptbr")

    def _nome_seguro(nome: str) -> str:
        limpo = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(nome))[:120]
        if not limpo.lower().endswith((".wav", ".mp3")):
            raise HTTPException(400, "só .wav ou .mp3")
        return limpo

    @app.on_event("startup")
    def _startup() -> None:
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=carregar, name="carregar-modelo", daemon=True).start()

    @app.get("/api/model-info")
    def model_info():
        with _model_lock:
            carregado = _model is not None
            sr = _model.sr if _model is not None else None
        corpo = {"loaded": carregado, "t3": T3_FILE, "s3gen": S3GEN_FILE, "device": DEVICE, "sample_rate": sr, "error": _load_error}
        return JSONResponse(corpo, status_code=200 if carregado else 503)

    @app.get("/get_reference_files")
    def referencias():
        return sorted(p.name for p in REFERENCE_DIR.glob("*") if p.suffix.lower() in (".wav", ".mp3"))

    @app.post("/upload_reference")
    async def upload_reference(files: list[UploadFile] = File(...)):
        enviados: list[str] = []
        erros: list[dict[str, str]] = []
        for arquivo in files:
            try:
                nome = _nome_seguro(arquivo.filename or "referencia.wav")
                conteudo = await arquivo.read(MAX_REFERENCE_BYTES + 1)
                if len(conteudo) > MAX_REFERENCE_BYTES:
                    raise HTTPException(413, "referência excede 8 MiB")
                if not conteudo:
                    raise HTTPException(400, "arquivo vazio")
                (REFERENCE_DIR / nome).write_bytes(conteudo)
                enviados.append(nome)
            except HTTPException as exc:
                erros.append({"filename": arquivo.filename or "", "error": str(exc.detail)})
        return {"message": f"Processed {len(files)} file(s).", "uploaded_files": enviados, "all_reference_files": referencias(), "errors": erros}

    @app.post("/tts")
    def tts(req: TtsRequest):
        with _model_lock:
            pronto = _model is not None
        if not pronto:
            raise HTTPException(503, "TTS engine model is not loaded")
        if req.voice_mode != "clone" or not req.reference_audio_filename:
            raise HTTPException(400, "este servidor só clona: voice_mode 'clone' com reference_audio_filename")
        referencia = REFERENCE_DIR / _nome_seguro(req.reference_audio_filename)
        if not referencia.is_file():
            raise HTTPException(404, "referência não encontrada")
        try:
            wav = sintetizar(req.text, referencia, req.language, req.exaggeration, req.cfg_weight, req.temperature, req.seed, req.split_text, req.chunk_size)
        except Exception as exc:  # noqa: BLE001
            log(f"síntese falhou: {type(exc).__name__}")
            raise HTTPException(500, f"síntese falhou: {type(exc).__name__}") from exc
        return Response(content=wav, media_type="audio/wav")

    @app.post("/api/unload")
    def unload():
        descarregar()
        return {"status": "unloaded"}

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="só monta o checkpoint no volume e sai")
    args = parser.parse_args()
    if args.download:
        ckpt = montar_checkpoint()
        log(f"checkpoint pronto em {ckpt}: {sorted(p.name for p in ckpt.iterdir())}")
        return
    import uvicorn

    uvicorn.run(criar_app(), host="0.0.0.0", port=PORT, log_level="warning")


if __name__ == "__main__":
    sys.exit(main())
