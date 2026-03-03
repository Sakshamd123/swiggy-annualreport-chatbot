from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    question: str = Field(..., description="User query about the Swiggy Annual Report")

class QueryResponse(BaseModel):
    answer: str
    confidence: str
    category: str
    source_pages: str
    context_snippet: Optional[str] = None
