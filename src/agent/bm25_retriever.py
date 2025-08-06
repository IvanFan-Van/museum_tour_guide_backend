from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from pathlib import Path
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import DirectoryLoader, TextLoader

stopword_corpus = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_fn(text: str):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopword_corpus]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


docs_folder = Path(
    r"D:\HKU\Inno Wing RA\UBC Exchange\code\output\Objectifying_China\docs"
)
loader = DirectoryLoader(
    str(docs_folder.absolute()),
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
docs = loader.load()
bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=preprocess_fn)
