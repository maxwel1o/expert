# Patch vllm-ascend compatibility issues for Qwen3-ASR
import types

def patch_ascend_mm_encoder_attention():
    try:
        from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention
        orig_init = AscendMMEncoderAttention.__init__

        def new_init(
            self,
            num_heads: int,
            head_size: int,
            scale: float | None = None,
            num_kv_heads: int | None = None,
            prefix: str = "",
            multimodal_config=None,
        ) -> None:
            orig_init(self, num_heads, head_size, scale, num_kv_heads, prefix)

        AscendMMEncoderAttention.__init__ = new_init
        print("Patched AscendMMEncoderAttention to accept multimodal_config")
    except Exception as e:
        print(f"Failed to patch AscendMMEncoderAttention: {e}")


def patch_get_vit_attn_backend():
    try:
        from vllm.model_executor.models.vision import get_vit_attn_backend as orig_fn

        def new_get_vit_attn_backend(
            head_size: int,
            dtype,
            attn_backend_override=None,
        ):
            return orig_fn(head_size, dtype)

        import vllm.model_executor.models.vision
        vllm.model_executor.models.vision.get_vit_attn_backend = new_get_vit_attn_backend
        print("Patched get_vit_attn_backend to accept attn_backend_override")
    except Exception as e:
        print(f"Failed to patch get_vit_attn_backend: {e}")


def patch_embed_text_input_ids():
    try:
        from vllm.model_executor.models.interfaces import SupportsMultiModal
        orig_method = SupportsMultiModal._embed_text_input_ids

        def new_embed_text_input_ids(
            self,
            input_ids,
            embed_input_ids,
            *,
            is_multimodal=None,
            handle_oov_mm_token=False,
        ):
            return orig_method(self, input_ids, embed_input_ids, is_multimodal=is_multimodal)

        SupportsMultiModal._embed_text_input_ids = new_embed_text_input_ids
        print("Patched SupportsMultiModal._embed_text_input_ids to accept handle_oov_mm_token")
    except Exception as e:
        print(f"Failed to patch _embed_text_input_ids: {e}")


patch_ascend_mm_encoder_attention()
patch_get_vit_attn_backend()
patch_embed_text_input_ids()
