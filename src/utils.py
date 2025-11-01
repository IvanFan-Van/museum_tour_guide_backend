import os
import re
from dotenv import load_dotenv, find_dotenv
from langchain_core.documents import Document
import colorlog
import logging

load_dotenv(find_dotenv())

gpt4o = None
gemma3_270m = None
zh_nlp = None
en_nlp = None
tts = None
logger = None


def get_gpt4o():
    global gpt4o
    if gpt4o is None:
        from langchain_openai import AzureChatOpenAI

        gpt4o = AzureChatOpenAI(
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYEMENT"],
            # max_tokens=16384,
        )
    return gpt4o


def get_gemma3_270m():
    global gemma3_270m
    if gemma3_270m is None:
        from langchain_ollama import ChatOllama

        gemma3_270m = ChatOllama(
            model="gemma3:270m",
            temperature=0.7,
        )
    return gemma3_270m


def get_tts():
    logger = get_logger()
    global tts
    if tts is None:
        import requests
        import base64

        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "AZURE_OPENAI_API_KEY is not set in environment variables."
            )
        tts_session = requests.Session()
        tts_session.headers.update(
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
        )

        def convert_text_to_speech_base64(text: str) -> str | None:
            endpoint = os.getenv("AZURE_OPENAI_TTS_ENDPOINT")
            if not endpoint:
                raise ValueError(
                    "AZURE_OPENAI_TTS_ENDPOINT is not set in environment variables."
                )
            try:
                response = tts_session.post(
                    endpoint,
                    json={
                        "input": text,
                        "voice": "alloy",
                        "model": "tts",
                    },
                )

                response.raise_for_status()

                return base64.b64encode(response.content).decode("utf-8")
            except requests.exceptions.RequestException as e:
                logger.error(f"Text-to-speech conversion failed. {e}")
                return None

        tts = convert_text_to_speech_base64

    return tts


def get_zh_nlp():
    global zh_nlp
    if zh_nlp is None:
        import spacy

        zh_nlp = spacy.load("zh_core_web_sm")
    return zh_nlp


def get_en_nlp():
    global en_nlp
    if en_nlp is None:
        import spacy

        en_nlp = spacy.load("en_core_web_sm")
    return en_nlp


def preprocess(text: str) -> list[str]:
    en_nlp = get_en_nlp()
    zh_nlp = get_zh_nlp()

    # remove image links
    image_link_pattern = r"!\[.*?\]\(.*?\)"
    text = re.sub(image_link_pattern, "", text)

    chinese_pattern = r"[\u4e00-\u9fff·]+"
    english_pattern = r"\w+"

    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    tokens = []
    for match in re.findall(chinese_pattern, text):
        if len(match) <= 2:
            tokens.append(match)
        else:
            doc = zh_nlp(match)
            tokens.extend([token.text for token in doc if not token.is_stop])

    for match in re.findall(english_pattern, text):
        doc = en_nlp(match)
        tokens.extend([token.text for token in doc if not token.is_stop])

    return tokens


def format_docs(docs: list[Document]) -> str:
    formatted = ""
    for idx, doc in enumerate(docs):
        formatted += f"DOC{idx + 1}: {doc.page_content}\n\n"
    return formatted


def get_logger():
    global logger
    logger = colorlog.getLogger("museum_tour_guide")

    if not logger.hasHandlers():
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(levelname)s%(reset)s:\t| %(message)s"
            )
        )

        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger
