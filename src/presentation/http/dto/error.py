from pydantic import BaseModel


class ErrorDetailDTO(BaseModel):
    code: str
    message: str


class ErrorResponseDTO(BaseModel):
    error: ErrorDetailDTO
