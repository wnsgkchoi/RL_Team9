from __future__ import annotations

import os
import logging
from typing import Dict, Any
from dataclasses import asdict

import uvicorn
from fastapi import FastAPI, HTTPException


from model.embedder.embedders.nvembed_v2.local import NVEmbedV2Local
from model.embedder.data_structure.signature import (
    EncodeRequest,
    EncodeListRequest,
    EncodeResponse,
    EncodeListResponse,
    InitiateResponse,
)
from model.embedder.data_structure.input import EncodingInput, EncodingParameters
import utils.directory as DirectoryUtils
import utils.file_io as FileIOUtils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NVEmbedV2_Server")






class NVEmbedV2Server:
    """FastAPI server for NV-Embed-v2 model encoding.

    Provides HTTP endpoints for model initialization and encoding operations.
    """

    def __init__(
        self,
        cuda_num: int = 0
    ) -> None:

        """Initialize the server.

        Args:
            cuda_num: CUDA device number to use
        """

        config_dir = os.path.join(DirectoryUtils.get_embedder_config_dir(), "nvembed_v2")
        config = FileIOUtils.read_yaml(os.path.join(config_dir, "_base.yaml"))

        self._host = config["remote_host"]
        self._port = config["remote_port"]
        self._cuda_num = cuda_num
        self._app = FastAPI(title="NV-Embed-v2 Encoding Server")
        self._model: NVEmbedV2Local = None
        self._register_routes()

        return



    def _register_routes(self):
        """Register FastAPI routes."""


        @self._app.post("/initiate")
        def initiate() -> Dict[str, Any]:
            """Initialize the model and return model information."""
            try:
                if self._model is None:
                    logger.info("Initializing NVEmbedV2Local model...")
                    self._model = NVEmbedV2Local(cuda_num=self._cuda_num)
                    self._model.initiate()
                    logger.info("Model initialized successfully")

                response = InitiateResponse(
                    model_name=self._model._model_name,
                    local_path=self._model._local_path,
                    vector_dim=self._model._vector_dim,
                    status="initialized",
                )
                return response.model_dump()

            except Exception as e:
                logger.error(f"Failed to initialize model: {e}")
                raise HTTPException(status_code=500, detail=str(e))



        @self._app.post("/encode_online")
        def encode_online(req: EncodeRequest) -> Dict[str, Any]:
            """Single encoding endpoint."""
            try:
                if self._model is None:
                    raise HTTPException(
                        status_code = 400,
                        detail = "Model not initialized. Call /initiate first."
                    )

                # Create EncodingInput
                enc_input = EncodingInput(
                    id              = req.id,
                    instruction     = req.instruction,
                    text            = req.text,
                    image_paths     = req.image_paths or [],
                )

                # Create EncodingParameters from dict
                parameters = None
                if req.parameters:
                    parameters = EncodingParameters(**req.parameters)

                # Run encoding
                enc_output, metadata = self._model.encode_online(enc_input, parameters)

                # Convert tensor to list (only if not writing to file)
                if parameters and parameters.write_file:
                    # File was written by local.py, return empty list
                    embeddings = []
                else:
                    embeddings = enc_output.tensors.tolist()

                # Create response
                response = EncodeResponse(
                    embeddings=embeddings,
                    metadata=asdict(metadata)
                )
                return response.model_dump()

            except Exception as e:
                logger.error(f"Encoding failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))



        @self._app.post("/encode_list_online")
        def encode_list_online(req: EncodeListRequest) -> Dict[str, Any]:
            """Batch encoding endpoint."""
            try:
                if self._model is None:
                    raise HTTPException(
                        status_code = 400,
                        detail = "Model not initialized. Call /initiate first."
                    )

                # Create list of EncodingInput
                enc_inputs = []
                for input_dict in req.inputs:
                    enc_input = EncodingInput(
                        id              = input_dict.get("id"),
                        instruction     = input_dict.get("instruction"),
                        text            = input_dict.get("text"),
                        image_paths     = input_dict.get("image_paths", []),
                    )
                    enc_inputs.append(enc_input)

                # Create EncodingParameters from dict
                parameters = None
                if req.parameters:
                    parameters = EncodingParameters(**req.parameters)

                # Run encoding
                enc_output, metadata = self._model.encode_list_online(enc_inputs, parameters)

                # Convert tensor to list (only if not writing to file)
                if parameters and parameters.write_file:
                    # File was written by local.py, return empty list
                    embeddings = []
                else:
                    embeddings = enc_output.tensors.tolist()

                # Create response
                response = EncodeListResponse(
                    embeddings=embeddings,
                    index=enc_output.index,
                    metadata=asdict(metadata)
                )
                return response.model_dump()

            except Exception as e:
                logger.error(f"Batch encoding failed: {e}")
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

    server = NVEmbedV2Server(cuda_num=0)
    server.run()
