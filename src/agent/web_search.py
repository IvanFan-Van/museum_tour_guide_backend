from concurrent.futures import ThreadPoolExecutor, as_completed

import wikipedia
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from wikipedia.exceptions import WikipediaException

from agent.prompts import SERACH_KEYWORDS_PROMPT
from agent.utils import llm


class Keywords(BaseModel):
    keywords: list[str] = Field(
        description="List of keywords extracted from the user's query."
    )


prompt = ChatPromptTemplate.from_template(SERACH_KEYWORDS_PROMPT)

llm_with_structure = llm.with_structured_output(Keywords)
keyword_generator = prompt | llm_with_structure


def get_keywords_with_retry(query: str) -> list[str]:
    keywords = Keywords.model_validate(
        keyword_generator.invoke({"user_query": query})
    ).keywords
    return keywords


def wiki_search(query: str) -> dict:
    """
    Search for summary of wikipedia page and return Documents

    Arguments:
    - query (str): user query

    Returns:
    list[Documents]
    """
    try:
        keywords = get_keywords_with_retry(query)
    except Exception as e:
        raise RuntimeError(f"failed to fetch key words: {e}") from e

    titles = set()
    for keyword in keywords:
        titles.update(wikipedia.search(keyword, results=1))

    def get_document(title):
        try:
            page = wikipedia.page(title=title)
            title = page.title
            content = wikipedia.summary(title)
            if not content:
                return None
            url = page.url
            image_lists = page.images[:2]
            images = [f"![]({image})" for image in image_lists]
            return Document(
                page_content=content + "\n" + "\n".join(images),
                metadata={"title": title, "url": url},
            )
        except WikipediaException:
            return None

    documents = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(get_document, title=title) for title in titles]

        for future in as_completed(futures):
            doc = future.result()
            if doc:
                documents.append(doc)

    return {"documents": documents, "keywords": keywords}
