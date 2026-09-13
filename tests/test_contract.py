import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import server


class ServerContract(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.patches = [
            patch.object(server, "REFERENCE_DIR", Path(self.directory.name)),
            patch.object(server, "_model", None),
            patch.object(server, "_load_error", None),
        ]
        for item in self.patches:
            item.start()
        self.app = server.criar_app()
        self.routes = {route.path: route.endpoint for route in self.app.routes}

    async def asyncTearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.directory.cleanup()

    async def test_readiness_tracks_model_load(self):
        health = self.routes["/api/model-info"]
        self.assertEqual(health().status_code, 503)
        with patch.object(server, "_model", SimpleNamespace(sr=24000)):
            self.assertEqual(health().status_code, 200)

    async def test_upload_and_clone_preserve_contract(self):
        from starlette.datastructures import UploadFile
        result = await self.routes["/upload_reference"]([
            UploadFile(filename="../../reference.wav", file=io.BytesIO(b"synthetic"))
        ])
        self.assertEqual(result["uploaded_files"], ["reference.wav"])
        self.assertEqual(self.routes["/get_reference_files"](), ["reference.wav"])
        with patch.object(server, "_model", SimpleNamespace(sr=24000)), patch.object(server, "sintetizar", return_value=b"RIFFsynthetic") as synth:
            response = self.routes["/tts"](server.TtsRequest(
                text="Synthetic test.", reference_audio_filename="reference.wav"
            ))
            self.assertEqual(response.media_type, "audio/wav")
            self.assertEqual(response.body, b"RIFFsynthetic")
            self.assertEqual(synth.call_args.args[1], Path(self.directory.name) / "reference.wav")

    async def test_missing_reference_and_unloaded_model_fail_explicitly(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as loading:
            self.routes["/tts"](server.TtsRequest(text="Synthetic"))
        self.assertEqual(loading.exception.status_code, 503)
        with patch.object(server, "_model", SimpleNamespace(sr=24000)):
            with self.assertRaises(HTTPException) as missing:
                self.routes["/tts"](server.TtsRequest(text="Synthetic", reference_audio_filename="absent.wav"))
            self.assertEqual(missing.exception.status_code, 404)

    async def test_reference_upload_size_is_bounded(self):
        from starlette.datastructures import UploadFile
        with patch.object(server, "MAX_REFERENCE_BYTES", 4):
            result = await self.routes["/upload_reference"]([
                UploadFile(filename="too-large.wav", file=io.BytesIO(b"12345"))
            ])
        self.assertEqual(result["uploaded_files"], [])
        self.assertEqual(len(result["errors"]), 1)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    async def test_chunking_preserves_words_and_punctuation(self):
        text = "Primeira frase. Segunda frase, com mais palavras! Outra frase?"
        chunks = server.dividir(text, 35)
        self.assertEqual(" ".join(chunks), text)
        self.assertTrue(all(len(chunk) <= 35 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
