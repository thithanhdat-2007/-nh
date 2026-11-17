"""HTTP client and SDXL workflow builder for local ComfyUI server."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

import requests

from assets_manifest import (
    CHECKPOINT_NAME,
    LORA_NAME,
    LORA_WEIGHT,
    NEGATIVE_PROMPT,
    AssetSpec,
)

DEFAULT_COMFY_URL = "http://127.0.0.1:8188"


class ComfyUIError(RuntimeError):
    """Raised when ComfyUI server is unreachable or fails a generation job."""


@dataclass(frozen=True)
class GeneratedImage:
    """Metadata and raw PNG bytes returned by a single ComfyUI run."""

    prompt_id: str
    seed: int
    filename: str
    subfolder: str
    image_bytes: bytes
    elapsed_sec: float


def build_sdxl_sumie_workflow(
    spec: AssetSpec,
    seed: int,
    steps: int = 35,
    cfg: float = 7.0,
    sampler_name: str = "euler_ancestral",
    scheduler: str = "normal",
) -> dict[str, Any]:
    """Build the ComfyUI API prompt graph using Animagine XL 3.1 + Sumi-e LoRA."""
    width, height = spec.gen_size
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": CHECKPOINT_NAME},
        },
        "2": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "lora_name": LORA_NAME,
                "strength_model": LORA_WEIGHT,
                "strength_clip": LORA_WEIGHT,
            },
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": spec.prompt,
                "clip": ["2", 1],
            },
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": NEGATIVE_PROMPT,
                "clip": ["2", 1],
            },
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
                "seed": int(seed),
                "steps": int(steps),
                "cfg": float(cfg),
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": 1.0,
            },
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["6", 0],
                "vae": ["1", 2],
            },
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["7", 0],
                "filename_prefix": f"VanKiem/{spec.id}",
            },
        },
    }


class ComfyClient:
    """Resilient synchronous REST client for ComfyUI API at 127.0.0.1:8188."""

    def __init__(self, base_url: str = DEFAULT_COMFY_URL, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())
        self.session = requests.Session()

    def _get_json(self, path: str, retries: int = 3) -> Any:
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                resp = self.session.get(f"{self.base_url}{path}", timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
        raise ComfyUIError(f"GET {path} failed after {retries} attempts: {last_exc}")

    def verify_server_and_models(self) -> dict[str, Any]:
        """Ensure ComfyUI is reachable and the required checkpoint and LoRA exist."""
        stats = self._get_json("/system_stats")
        ckpts = self._get_json("/models/checkpoints")
        loras = self._get_json("/models/loras")

        if CHECKPOINT_NAME not in ckpts:
            raise ComfyUIError(
                f"Required checkpoint '{CHECKPOINT_NAME}' not found on server. "
                f"Available: {ckpts}"
            )
        if LORA_NAME not in loras:
            raise ComfyUIError(
                f"Required LoRA '{LORA_NAME}' not found on server. Available: {loras}"
            )
        return stats

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        """Submit a workflow prompt and return its ComfyUI prompt_id."""
        payload = {"prompt": workflow, "client_id": self.client_id}
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = self.session.post(
                    f"{self.base_url}/prompt", json=payload, timeout=self.timeout
                )
                if resp.status_code != 200:
                    raise ComfyUIError(
                        f"ComfyUI /prompt rejected request ({resp.status_code}): {resp.text}"
                    )
                data = resp.json()
                if data.get("node_errors"):
                    raise ComfyUIError(
                        f"ComfyUI node validation errors: {data['node_errors']}"
                    )
                return str(data["prompt_id"])
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(2.0 * (attempt + 1))
        raise ComfyUIError(f"POST /prompt failed after 3 attempts: {last_exc}")

    def wait_for_result(
        self,
        prompt_id: str,
        poll_interval: float = 1.0,
        max_wait_sec: float = 600.0,
    ) -> dict[str, Any]:
        """Poll /history/{prompt_id} until completion or error."""
        deadline = time.monotonic() + max_wait_sec
        while time.monotonic() < deadline:
            history = self._get_json(f"/history/{prompt_id}")
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise ComfyUIError(
                        f"ComfyUI execution error for {prompt_id}: {status.get('messages')}"
                    )
                if status.get("completed", False) or entry.get("outputs"):
                    return entry
            time.sleep(poll_interval)
        raise ComfyUIError(
            f"Timed out waiting {max_wait_sec}s for ComfyUI prompt {prompt_id}"
        )

    def download_image(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        """Fetch generated image bytes from ComfyUI /view endpoint."""
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = self.session.get(
                    f"{self.base_url}/view",
                    params={
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": folder_type,
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.content
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
        raise ComfyUIError(f"GET /view failed after 3 attempts: {last_exc}")

    def generate_variant(
        self,
        spec: AssetSpec,
        seed: int,
        steps: int = 35,
        cfg: float = 7.0,
        sampler_name: str = "euler_ancestral",
        scheduler: str = "normal",
    ) -> GeneratedImage:
        """Run one ComfyUI generation for an AssetSpec and return its PNG bytes."""
        started = time.monotonic()
        workflow = build_sdxl_sumie_workflow(
            spec=spec,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
        )
        prompt_id = self.queue_prompt(workflow)
        history_entry = self.wait_for_result(prompt_id)
        outputs = history_entry.get("outputs", {})
        save_out = outputs.get("8", {})
        images = save_out.get("images", [])
        if not images:
            raise ComfyUIError(
                f"Prompt {prompt_id} finished without SaveImage output: {outputs}"
            )
        img_meta = images[0]
        raw_bytes = self.download_image(
            filename=img_meta["filename"],
            subfolder=img_meta.get("subfolder", ""),
            folder_type=img_meta.get("type", "output"),
        )
        return GeneratedImage(
            prompt_id=prompt_id,
            seed=seed,
            filename=img_meta["filename"],
            subfolder=img_meta.get("subfolder", ""),
            image_bytes=raw_bytes,
            elapsed_sec=round(time.monotonic() - started, 2),
        )
