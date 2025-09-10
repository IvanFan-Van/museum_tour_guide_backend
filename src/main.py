import io
import warnings
from typing import TypedDict

import soundfile as sf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from speechbrain.inference.TTS import FastSpeech2
from speechbrain.inference.vocoders import HIFIGAN

from src.agent.graph import graph

warnings.filterwarnings("ignore")

origins = ["*"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(TypedDict):
    query: str


@app.post("/invoke")
def invoke(req: QueryRequest):
    messages = [HumanMessage(content=req["query"])]
    response = graph.invoke({"messages": messages})
    # print(response)
    return response.get("generation", None)


# Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
fastspeech2 = FastSpeech2.from_hparams(
    source="speechbrain/tts-fastspeech2-ljspeech",
    savedir="pretrained_models/tts-fastspeech2-ljspeech",
)
hifi_gan = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech",
    savedir="pretrained_models/tts-hifigan-ljspeech",
)


@app.post("/tts")
def tts(req: QueryRequest):
    input_text = req["query"]
    if fastspeech2 is None:
        return {"status": "error", "error": "TTS model not loaded"}, 500

    if hifi_gan is None:
        return {"status": "error", "error": "Vocoder model not loaded"}, 500

    mel_output, durations, pitch, energy = fastspeech2.encode_text(
        [input_text],
        pace=1.0,  # scale up/down the speed
        pitch_rate=1.0,  # scale up/down the pitch
        energy_rate=1.0,  # scale up/down the energy
    )

    print(1)
    waveforms = hifi_gan.decode_batch(mel_output)
    print(2)

    wav_buffer = io.BytesIO()
    sf.write(
        wav_buffer, waveforms.squeeze().cpu().numpy(), samplerate=22050, format="WAV"
    )
    print(3)

    wav_buffer.seek(0)

    return StreamingResponse(wav_buffer, media_type="audio/wav")
