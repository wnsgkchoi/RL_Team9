from __future__ import annotations

import os
import logging
from typing import Dict, Any
from dataclasses import asdict

import uvicorn
from fastapi import FastAPI, HTTPException


from model.mllm.mllms.qwen2_5_vl_7b.local import Qwen2_5_VL_Local
from model.mllm.data_structure.signature import (
    InferRequest,
    InferListRequest,
    InferResponse,
    InferListResponse,
    InitiateResponse,
)
from model.mllm.data_structure.input import GenerationInput, GenerationParameters
import utils.directory as DirectoryUtils
import utils.file_io as FileIOUtils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Qwen2_5_VL_Server")






class Qwen2_5_VL_Server:
    """FastAPI server for Qwen2.5-VL-7B model inference.

    Provides HTTP endpoints for model initialization and inference operations.
    """

    def __init__(
        self,
    ) -> None:

        """Initialize the server.

        Args:
            host: Host address to bind to
            port: Port number to bind to
        """

        config_dir = os.path.join(DirectoryUtils.get_mllm_config_dir(), "qwen2_5_vl_7b")
        config = FileIOUtils.read_yaml(os.path.join(config_dir, "_base.yaml"))

        self._host = config["remote_host"]
        self._port = config["remote_port"]
        self._app = FastAPI(title="Qwen2.5-VL-7B Inference Server")
        self._model: Qwen2_5_VL_Local = None
        self._register_routes()

        return



    def _register_routes(self):
        """Register FastAPI routes."""


        @self._app.post("/initiate")
        def initiate() -> Dict[str, Any]:
            """Initialize the model and return model information."""
            try:
                if self._model is None:
                    logger.info("Initializing Qwen2_5_VL_Local model...")
                    self._model = Qwen2_5_VL_Local()
                    self._model.initiate()
                    logger.info("Model initialized successfully")

                response = InitiateResponse(
                    model_name=self._model._model_name,
                    local_path=self._model._local_path,
                    status="initialized",
                )
                return response.model_dump()

            except Exception as e:
                logger.error(f"Failed to initialize model: {e}")
                raise HTTPException(status_code=500, detail=str(e))



        @self._app.post("/infer")
        def infer(req: InferRequest) -> Dict[str, Any]:
            """Single inference endpoint."""
            try:
                if self._model is None:
                    raise HTTPException(
                        status_code = 400,
                        detail = "Model not initialized. Call /initiate first."
                    )

                # Create GenerationInput
                gen_input = GenerationInput(
                    text_prompt     = req.text_prompt,
                    system_prompt   = req.system_prompt,
                    image_paths     = req.image_paths or [],
                    video_paths     = req.video_paths or [],
                )

                # Create GenerationParameters from dict
                parameters = None
                if req.parameters:
                    parameters = GenerationParameters(**req.parameters)

                # Run inference
                gen_output, metadata = self._model.infer(gen_input, parameters)

                # Create response
                response = InferResponse(
                    text=gen_output.text, metadata=asdict(metadata)
                )
                return response.model_dump()

            except Exception as e:
                logger.error(f"Inference failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))



        @self._app.post("/infer_list")
        def infer_list(req: InferListRequest) -> Dict[str, Any]:
            """Batch inference endpoint."""
            try:
                if self._model is None:
                    raise HTTPException(
                        status_code = 400,
                        detail = "Model not initialized. Call /initiate first."
                    )

                # Create list of GenerationInput
                gen_inputs = []
                for input_dict in req.inputs:
                    gen_input = GenerationInput(
                        text_prompt     = input_dict.get("text_prompt"),
                        system_prompt   = input_dict.get("system_prompt"),
                        image_paths     = input_dict.get("image_paths", []),
                        video_paths     = input_dict.get("video_paths", []),
                    )
                    gen_inputs.append(gen_input)

                # Create GenerationParameters from dict
                parameters = None
                if req.parameters:
                    parameters = GenerationParameters(**req.parameters)

                # Run inference
                gen_outputs, metadata = self._model.infer_list(gen_inputs, parameters)

                # Create response
                outputs = [output.text for output in gen_outputs]
                response = InferListResponse(outputs=outputs, metadata=asdict(metadata))
                return response.model_dump()

            except Exception as e:
                logger.error(f"Batch inference failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))



        @self._app.get("/health")
        def health() -> Dict[str, str]:
            """Health check endpoint."""
            return {
                "status": "healthy",
                "model_loaded": self._model is not None,
            }

    def run(self):
        """Run the server."""
        logger.info(f"Starting server on {self._host}:{self._port}")
        uvicorn.run(self._app, host=self._host, port=self._port)



if __name__ == "__main__":

    server = Qwen2_5_VL_Server()
    server.run()
