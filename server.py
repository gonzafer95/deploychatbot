from main import chain

from fastapi import FastAPI
from langserve import add_routes


app = FastAPI(
  title="Chatbot Promptior",
  version="1.0",
  description="Un chatbot para la prueba de promptior ai engineer",
)

add_routes(app, chain)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=process.env.PORT)
