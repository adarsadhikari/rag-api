from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class ModelName(str, Enum):
    GPT_OSS_120B='GPT-oss-120b'
    GPT_OSS_20B='GPT-oss-20b'
    Deepseek='DeepSeek-V3.2'


class QueryInput(BaseModel):
    question:str
    session_id:str=Field(default=None)
    model:ModelName=Field(default=ModelName.GPT_OSS_120B)

class QueryResponse(BaseModel):
    answer:str
    session_id:str
    model:ModelName

class DocumentInfo(BaseModel):
    id:int
    filename:str
    upload_timestamp:datetime

class DeleteFileRequest(BaseModel):
    file_id:int