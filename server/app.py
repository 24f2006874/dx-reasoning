# server/app.py
from openenv.core.env_server.http_server import create_app
from .dx_reasoning_environment import DxReasoningEnvironment
try:
    from ..models import DxAction, DxObservation
except ImportError:
    from models import DxAction, DxObservation
import uvicorn

app = create_app(DxReasoningEnvironment, DxAction, DxObservation, env_name="dx_reasoning")

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()