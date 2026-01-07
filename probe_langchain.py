
try:
    import langchain
    print(f"langchain version: {langchain.__version__}")
except ImportError:
    print("langchain not found")

try:
    from langchain.retrievers import document_compressors
    print("Found langchain.retrievers.document_compressors")
except ImportError as e:
    print(f"Error importing langchain.retrievers.document_compressors: {e}")

try:
    from langchain_core.documents import BaseDocumentCompressor
    print("Found BaseDocumentCompressor in langchain_core.documents")
except ImportError:
    print("Not found in langchain_core.documents")

try:
    from langchain.retrievers.document_compressors import BaseDocumentCompressor
    print("Found BaseDocumentCompressor in langchain.retrievers.document_compressors")
except ImportError:
    print("Not found in langchain.retrievers.document_compressors")
