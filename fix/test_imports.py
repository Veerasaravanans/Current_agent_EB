try:
    from langchain_ollama import ChatOllama
    from pydantic import BaseModel, Field
    print('Imports successful')
except Exception as e:
    print(f'Import error: {e}')
