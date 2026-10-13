"""Alternative embedding backbones for the proactive-gate ablation.

The surge gate and its d = 0.03 threshold were tuned on Qwen3-VL-Embedding-2B.
Whether that threshold is a property of the *method* or an artefact of that one
model is only answerable by running the identical controller on other encoders,
so this module supplies the minimum interface the frame-mean matching path needs:

    query_vectors(texts, instruction=...) -> ([vector], seconds)
    encode_frame(image)                   -> normalized vector

Every backend returns L2-normalized vectors in a shared image/text space, which
is what the surge rule assumes -- it compares cosine similarity of a frame window
against frozen proposal-text vectors, so a backend whose image and text towers do
not share a space cannot host the gate at all, and that is itself a finding.

Two-tower encoders (CLIP, SigLIP, jina-clip) support only the frame-mean path:
they have no decoder to rerun over a multi-frame window. The Qwen2-VL-family
embedders (GME, Ops-MM, VLM2Vec) also implement cache A
(`encode_joint_images_cached_vit`), the main method's path.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from PIL import Image


class DualEncoderBackend:
    """CLIP / SigLIP / jina-clip style two-tower encoders."""

    def __init__(self, model_path: str, device: torch.device, dtype: torch.dtype):
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.dtype = dtype
        self.model_path = str(model_path)
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        from transformers.models.clip import modeling_clip
        if not hasattr(modeling_clip, "clip_loss"):
            # jina-clip's remote code imports this training-only helper, which
            # transformers 5.x removed; restore the original definition.
            def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
                target = torch.arange(len(similarity), device=similarity.device)
                return (torch.nn.functional.cross_entropy(similarity, target)
                        + torch.nn.functional.cross_entropy(similarity.t(), target)) / 2.0
            modeling_clip.clip_loss = clip_loss
        if "jina" in self.model_path.lower():
            self.model = self._load_jina(self.model_path).to(dtype)
        else:
            self.model = AutoModel.from_pretrained(
                self.model_path, torch_dtype=dtype, trust_remote_code=True
            )
        self.model = self.model.to(device)
        self.model.eval()

    @staticmethod
    def _load_jina(model_path: str):
        """Build jina-clip with real storage, then load its checkpoint.

        from_pretrained in transformers 5.x builds the model without storage and
        fills it from the checkpoint, so every non-persistent buffer that jina's
        remote code computes in __init__ is left uninitialised: the vision RoPE
        tables come out NaN (every image embedding NaN) and the text tower's
        rotary inv_freq comes out as garbage. Constructing from the config
        computes those buffers; the checkpoint then supplies every weight.
        """
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_config(config, trust_remote_code=True)
        result = model.load_state_dict(load_file(f"{model_path}/model.safetensors"), strict=False)
        missing = [k for k in result.missing_keys if "lora_dropout_mask" not in k]
        if missing or result.unexpected_keys:
            raise RuntimeError(f"jina-clip checkpoint mismatch: missing={missing[:5]} "
                               f"unexpected={result.unexpected_keys[:5]}")
        return model

    @property
    def representation_kind(self) -> str:
        return "single_vector"

    @torch.inference_mode()
    def query_vectors(self, texts, *, instruction=None):
        if not texts:
            raise ValueError("query_vectors requires at least one text")
        started = time.perf_counter()
        # The instruction is prepended rather than passed as a separate field:
        # a two-tower text encoder has no instruction slot, and dropping it would
        # silently change what the proposal vector means relative to ours.
        prompts = [f"{instruction} {text}".strip() if instruction else str(text) for text in texts]
        inputs = self.processor(
            text=prompts, padding="max_length", truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        features = self._project(self.model.get_text_features(**inputs), "text_projection")
        vectors = [F.normalize(row.float(), dim=-1).cpu().reshape(-1) for row in features]
        return vectors, time.perf_counter() - started

    @torch.inference_mode()
    def encode_frame(self, frame: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=[frame], return_tensors="pt")
        inputs = {k: v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device)
                  for k, v in inputs.items() if torch.is_tensor(v)}
        features = self._project(self.model.get_image_features(**inputs), "visual_projection")
        return F.normalize(features.float()[0], dim=-1).cpu().reshape(-1)

    def _project(self, output, projection_name):
        """Return projected tower features as a tensor.

        Some transformers versions return a bare tensor from get_*_features and
        others return a ModelOutput whose pooled state still needs the projection
        that maps the tower into the shared image/text space.  Iterating a
        ModelOutput yields its *keys*, so the difference has to be resolved here
        rather than downstream.
        """
        if torch.is_tensor(output):
            return output
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            pooled = output.last_hidden_state[:, 0]
        # This transformers version already applies the tower projection inside
        # get_*_features and only wraps the result, so projecting again would be
        # wrong.  Project only when the pooled width still differs from the
        # shared-space width, which is the case where it genuinely has not been.
        target = getattr(self.model.config, "projection_dim", None)
        if target is not None and int(pooled.shape[-1]) == int(target):
            return pooled
        projection = getattr(self.model, projection_name, None)
        if projection is None:
            return pooled
        expected = getattr(projection, "in_features", None)
        if expected is not None and int(pooled.shape[-1]) != int(expected):
            return pooled
        return projection(pooled)


class GmeQwen2VLBackend:
    """GME-Qwen2-VL-2B: a decoder-based multimodal embedder, same family as ours.

    This is the informative comparison: it shares the architecture class with
    Qwen3-VL-Embedding-2B, so a threshold that transfers here but not to CLIP
    would say the gate depends on decoder-style embeddings rather than on one
    specific checkpoint.
    """

    def __init__(self, model_path: str, device: torch.device, dtype: torch.dtype):
        # transformers 5.x dropped AutoModelForVision2Seq; the checkpoint declares
        # Qwen2VLForConditionalGeneration, which AutoModelForImageTextToText maps.
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        self.dtype = dtype
        # trust_remote_code is deliberately off: GME's custom module pins
        # transformers<4.52 and this repo runs a 5.x fork.  Its embedding is
        # last-token pooling over Qwen2-VL's final hidden state, which _embed
        # below reproduces directly, so the stock architecture is enough.
        self.processor = AutoProcessor.from_pretrained(str(model_path))
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_path), dtype=dtype
        ).to(device)
        self.model.eval()

    @property
    def representation_kind(self) -> str:
        return "single_vector"

    def _pool_last(self, hidden, attention_mask):
        lengths = attention_mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]

    # Copied verbatim from the checkpoint's gme_inference.py.  A generic chat
    # template gives a different final token, and since the embedding IS the
    # final token's hidden state, using the wrong template collapses every
    # vector onto near-identical directions.
    _TEMPLATE = (
        "<|im_start|>system\n{instruction}<|im_end|>\n"
        "<|im_start|>user\n{body}<|im_end|>\n"
        "<|im_start|>assistant\n<|endoftext|>"
    )
    _DEFAULT_INSTRUCTION = "You are a helpful assistant."

    @torch.inference_mode()
    def _embed(self, text=None, image=None, instruction=None):
        body = ""
        if image is not None:
            body += "<|vision_start|><|image_pad|><|vision_end|>"
        if text:
            body += text
        prompt = self._TEMPLATE.format(
            instruction=instruction or self._DEFAULT_INSTRUCTION, body=body
        )
        inputs = self.processor(
            text=[prompt],
            images=[image] if image is not None else None,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        # transformers 5.x needs mm_token_type_ids alongside image_grid_thw to
        # build multimodal RoPE, so the processor's tensors are forwarded as a
        # set rather than cherry-picked.
        outputs = self.model.model(
            **{
                key: value
                for key, value in inputs.items()
                if key in (
                    "input_ids", "attention_mask", "pixel_values",
                    "image_grid_thw", "mm_token_type_ids",
                )
            }
        )
        pooled = self._pool_last(outputs.last_hidden_state, inputs["attention_mask"])
        return F.normalize(pooled.float(), dim=-1)

    # ---- cache A for Qwen2-VL-family embedders ------------------------------
    # The same mechanism as the default encoder's encode_joint_images_cached_vit:
    # each frame's ViT output is computed once and cached by (video, position);
    # every poll rebuilds the multi-image token sequence, scatters the cached
    # visual tokens into it and reruns only the decoder, then reads the vector
    # the way this embedder reads a single image. Qwen2-VL has no DeepStack, so
    # the cached entry is just the merged visual tokens.
    def _window_prompt(self, n_images: int) -> str:
        body = "<|vision_start|><|image_pad|><|vision_end|>" * int(n_images)
        return self._TEMPLATE.format(instruction=self._DEFAULT_INSTRUCTION, body=body)

    @torch.inference_mode()
    def encode_joint_images_cached_vit(self, frames, frame_keys, feature_cache, cache_size):
        if not frames or len(frames) != len(frame_keys):
            raise ValueError("cached-ViT encoding needs one key per non-empty frame")
        inputs = self.processor(text=[self._window_prompt(len(frames))], images=list(frames),
                                padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        grid_thw = inputs["image_grid_thw"]
        pixel_chunks = torch.split(inputs["pixel_values"], [int(v) for v in grid_thw.prod(-1).tolist()], dim=0)
        core = self.model.model
        entries, hits, vit_calls, vit_seconds = [], 0, 0, 0.0
        for key, grid, pixels in zip(frame_keys, grid_thw, pixel_chunks):
            grid_cpu = grid.detach().cpu()
            cached = feature_cache.pop(key, None)
            if cached is not None and torch.equal(cached["grid"], grid_cpu):
                feature_cache[key] = cached
                entries.append(cached)
                hits += 1
                continue
            started = time.perf_counter()
            out = core.get_image_features(pixels.to(self.dtype), grid.unsqueeze(0), return_dict=True)
            pooled = out.pooler_output[0] if isinstance(out.pooler_output, (list, tuple)) else out.pooler_output
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            vit_seconds += time.perf_counter() - started
            cached = {"grid": grid_cpu, "pooled": pooled.detach()}
            feature_cache[key] = cached
            while len(feature_cache) > int(cache_size):
                feature_cache.popitem(last=False)
            entries.append(cached)
            vit_calls += 1
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        inputs_embeds = core.get_input_embeddings()(input_ids)
        image_embeds = torch.cat([e["pooled"] for e in entries], dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = core.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        position_ids = core.compute_3d_position_ids(
            input_ids=input_ids, inputs_embeds=inputs_embeds, image_grid_thw=grid_thw,
            video_grid_thw=None, attention_mask=attention_mask, past_key_values=None,
            mm_token_type_ids=inputs.get("mm_token_type_ids"))
        started = time.perf_counter()
        out = core.language_model(input_ids=None, position_ids=position_ids, attention_mask=attention_mask,
                                  inputs_embeds=inputs_embeds, use_cache=False, return_dict=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        decoder_seconds = time.perf_counter() - started
        vector = F.normalize(self._pool_last(out.last_hidden_state, attention_mask).float(), dim=-1)[0]
        return vector.cpu().reshape(-1), {
            "proposal_vit_feature_encode_calls_this_window": vit_calls,
            "proposal_vit_feature_cache_hits_this_window": hits,
            "proposal_vit_feature_seconds_this_window": vit_seconds,
            "proposal_joint_decoder_seconds_this_window": decoder_seconds,
        }

    def query_vectors(self, texts, *, instruction=None):
        if not texts:
            raise ValueError("query_vectors requires at least one text")
        started = time.perf_counter()
        vectors = []
        for text in texts:
            # Query side carries the instruction in the system slot, exactly as
            # GME does; the document side always uses its default instruction.
            vectors.append(
                self._embed(text=str(text), instruction=instruction)[0].cpu().reshape(-1)
            )
        return vectors, time.perf_counter() - started

    def encode_frame(self, frame: Image.Image) -> torch.Tensor:
        return self._embed(image=frame)[0].cpu().reshape(-1)


_BACKENDS = {
    "clip": DualEncoderBackend,
    "siglip": DualEncoderBackend,
    "jina_clip": DualEncoderBackend,
    "gme_qwen2vl": GmeQwen2VLBackend,
    # Ops-MM-Embedding-2B uses byte-identical prompting and last-token pooling
    # to GME (checked against its ops_mm_embedding_v1.py), so it shares the
    # backend rather than getting a near-duplicate class.
    "opsmm": GmeQwen2VLBackend,
}


def build_embedding_backend(name: str, model_path: str, device, dtype):
    key = str(name).lower().replace("-", "_")
    if key not in _BACKENDS:
        raise ValueError(
            f"Unknown embedding_backend={name!r}; known: {sorted(_BACKENDS)} "
            "(or 'qwen3vl' for the default encoder, which is not built here)."
        )
    return _BACKENDS[key](model_path, device, dtype)


class VLM2VecQwen2VLBackend(GmeQwen2VLBackend):
    """VLM2Vec-Qwen2VL-2B: a LoRA adapter over Qwen2-VL-2B-Instruct.

    Same architecture class as GME and Ops-MM, so the pooling is inherited, but
    two things differ and both matter.  The weights are a LoRA that has to be
    merged onto the stock base, and the prompt is NOT a chat template: VLM2Vec
    feeds raw text carrying the image tokens, so reusing GME's im_start/im_end
    wrapper would change the final token that the embedding is read from.
    """

    _IMAGE_TOKENS = "<|vision_start|><|image_pad|><|vision_end|>"

    def __init__(self, model_path: str, device, dtype):
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        base = "models/Qwen2-VL-2B-Instruct"
        self.device = device
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(base)
        model = AutoModelForImageTextToText.from_pretrained(base, dtype=dtype)
        model = PeftModel.from_pretrained(model, str(model_path))
        self.model = model.merge_and_unload().to(device)
        self.model.eval()

    def _window_prompt(self, n_images: int) -> str:
        return (self._IMAGE_TOKENS * int(n_images)) + " Represent the given image."

    @torch.inference_mode()
    def _embed(self, text=None, image=None, instruction=None):
        body = self._IMAGE_TOKENS if image is not None else ""
        if instruction and text:
            body += f" {instruction} {text}"
        elif text:
            body += f" {text}"
        elif image is not None:
            body += " Represent the given image."
        inputs = self.processor(
            text=[body.strip()],
            images=[image] if image is not None else None,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        outputs = self.model.model(
            **{
                key: value
                for key, value in inputs.items()
                if key in (
                    "input_ids", "attention_mask", "pixel_values",
                    "image_grid_thw", "mm_token_type_ids",
                )
            }
        )
        pooled = self._pool_last(outputs.last_hidden_state, inputs["attention_mask"])
        return F.normalize(pooled.float(), dim=-1)


_BACKENDS["vlm2vec"] = VLM2VecQwen2VLBackend
