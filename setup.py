import argparse
import json
from pathlib import Path

from chromadb import PersistentClient


def main():
    """
    Usage:
        python backend/setup.py [--reset]

    This script sets up the ChromaDB vector database with documents from the
    'data/Objectifying_China/docs' directory.

    By default, it will update the existing 'museum_knowledge_base' collection
    with the latest documents. This is an 'upsert' operation, meaning it will
    add new documents and update existing ones based on their IDs.

    Arguments:
        --reset     If this flag is provided, the script will first delete the
                    'museum_knowledge_base' collection if it exists, and then
                    recreate it before adding the documents. This is useful
                    for a clean, fresh start.
    """
    parser = argparse.ArgumentParser(
        description="Setup ChromaDB with museum knowledge base"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the ChromaDB database before setup",
    )
    args = parser.parse_args()

    client = PersistentClient(path="chroma_db")
    collection_name = "museum_knowledge_base"

    if args.reset:
        print("Resetting ChromaDB database...")
        try:
            client.delete_collection(name=collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception as e:
            print(f"No existing collection to delete: {e}")
        collection = client.create_collection(name=collection_name)
        print(f"Created new collection: {collection_name}")
    else:
        print("Using existing ChromaDB database...")
        collection = client.get_or_create_collection(name=collection_name)

    docs_path = Path("data/Objectifying_China/docs")

    documents = []
    ids = []
    metadatas = []
    for filepath in docs_path.glob("*.json"):
        with open(filepath.absolute(), "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading {filepath}")
                continue

            documents.append(data["documents"])
            ids.append(data["ids"])
            data["metadata"].pop("description", None)
            data["metadata"].pop("images", None)
            metadatas.append(data["metadata"])

    if not ids:
        print("No documents found to add to the database.")
        return

    print(f"Attempting update {len(ids)} documents to the collection...")
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"After setup, the collection has the following stats: {collection.count()} documents.")


if __name__ == "__main__":
    main()
