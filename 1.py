import chromadb

c = chromadb.PersistentClient(path="storage/mh_equipment_chroma_db")
print(c.get_collection("langchain").configuration)
