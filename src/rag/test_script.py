from langchain_text_splitters import MarkdownHeaderTextSplitter
from pathlib import Path

text = Path("data/monster.md").read_text(encoding="utf-8")
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "EnemyName"), ("##", "_section")]
)
docs = splitter.split_text(text)

for doc in docs[:6]:
    print(doc.metadata)
    print(doc.page_content[:40])
    print("---")