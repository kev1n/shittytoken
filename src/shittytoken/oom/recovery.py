"""
OOM recovery orchestration.

CRITICAL INVARIANT: write_oom_event() MUST complete before ANY destructive
action (destroy_fn, provision_fn). This is enforced structurally by the
sequential await ordering in recover() — not by assertion or documentation.

Step order is:
  1. classify_oom         (sync, no side-effects)
  2. await write_oom_event  ← first await; captures event_id
  3. await prior_oom_resolutions
  4. await reason_about_oom
  5. build new Configuration
  6. await write_configuration
  [try]
  7. await destroy_fn
  8. await provision_fn
  [finally]
  9. await update_oom_outcome   ← always called
"""

from __future__ import annotations

import structlog
import anthropic

from ..knowledge.client import KnowledgeGraph
from ..knowledge.schema import Configuration
from .detector import classify_oom, OOMClassification, OOMType
from .reasoner import reason_about_oom, OOMContext, OOMReasoningError

logger = structlog.get_logger()


class OOMRecovery:
    """
    Orchestrates OOM recovery in strict step order.

    CRITICAL INVARIANT: write_oom_event() completes before ANY destructive
    action. This is enforced by the sequential await structure — not by
    documentation or assertion.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        anthropic_client: anthropic.AsyncAnthropic,
        model: str = "claude-opus-4-6",
    ) -> None:
        self._kg = kg
        self._anthropic_client = anthropic_client
        self._model = model

    async def recover(
        self,
        instance_id: str,
        gpu_model_name: str,
        gpu_vram_gb: int,
        gpu_memory_free_gb: float,
        model_id: str,
        model_params_b: float,
        model_active_params_b: float,
        current_config: Configuration,
        raw_error: str,
        log_context: str,
        destroy_fn,    # async callable(instance_id: str) -> None
        provision_fn,  # async callable(config: Configuration) -> str | None
    ) -> tuple[bool, Configuration | None]:
        """
        Executes the recovery flow in strict order.

        Returns (succeeded, new_config) where new_config is None on failure.

        If reason_about_oom raises OOMReasoningError:
        - Log the error at WARNING level
        - Call update_oom_outcome(succeeded=False)
        - Return (False, None)

        The try/finally around destroy_fn/provision_fn guarantees
        update_oom_outcome is called even when those callables raise.
        """
        # ------------------------------------------------------------------
        # STEP 1: classify (sync, no side-effects)
        # ------------------------------------------------------------------
        classification: OOMClassification = classify_oom(raw_error, log_context)
        oom_type_str = classification.oom_type.value

        logger.info(
            "oom_recovery.classified",
            instance_id=instance_id,
            oom_type=oom_type_str,
            confidence=classification.confidence,
            matched_pattern=classification.matched_pattern,
        )

        # ------------------------------------------------------------------
        # STEP 2: write_oom_event — FIRST AWAIT, BEFORE ANY DESTRUCTIVE ACTION
        # ------------------------------------------------------------------
        event_id: str = await self._kg.write_oom_event(
            config_id=current_config.config_id,
            gpu_model_name=gpu_model_name,
            error_type=oom_type_str,
            error_message=raw_error,
            error_phase=oom_type_str,
            gpu_memory_free_gb=gpu_memory_free_gb,
            gpu_memory_total_gb=float(gpu_vram_gb),
        )

        logger.info(
            "oom_recovery.event_written",
            event_id=event_id,
            instance_id=instance_id,
        )

        # ------------------------------------------------------------------
        # STEP 3: fetch prior resolutions for context
        # ------------------------------------------------------------------
        resolutions = await self._kg.prior_oom_resolutions(
            gpu_model_name=gpu_model_name,
            error_type=oom_type_str,
        )

        # ------------------------------------------------------------------
        # STEP 4: ask the LLM for a proposed fix
        # ------------------------------------------------------------------
        ctx = OOMContext(
            gpu_model_name=gpu_model_name,
            gpu_vram_gb=gpu_vram_gb,
            gpu_memory_free_gb=gpu_memory_free_gb,
            model_id=model_id,
            params_b=model_params_b,
            active_params_b=model_active_params_b,
            current_config={
                "tensor_parallel_size": current_config.tensor_parallel_size,
                "max_model_len": current_config.max_model_len,
                "gpu_memory_utilization": current_config.gpu_memory_utilization,
                "quantization": current_config.quantization,
                "kv_cache_dtype": current_config.kv_cache_dtype,
                "max_num_seqs": current_config.max_num_seqs,
                "enable_prefix_caching": current_config.enable_prefix_caching,
                "enforce_eager": current_config.enforce_eager,
            },
            oom_type=oom_type_str,
            raw_error=raw_error,
            prior_resolutions=resolutions,
        )

        try:
            proposed_config_dict = await reason_about_oom(
                ctx=ctx,
                anthropic_client=self._anthropic_client,
                model=self._model,
            )
        except OOMReasoningError as exc:
            logger.warning(
                "oom_recovery.reasoning_failed",
                event_id=event_id,
                instance_id=instance_id,
                error=str(exc),
            )
            await self._kg.update_oom_outcome(
                event_id=event_id,
                succeeded=False,
                resolution_config_id=None,
            )
            return (False, None)

        # ------------------------------------------------------------------
        # STEP 5: build new Configuration by merging proposed fields
        # ------------------------------------------------------------------
        try:
            new_config = Configuration(
                tensor_parallel_size=proposed_config_dict.get(
                    "tensor_parallel_size", current_config.tensor_parallel_size
                ),
                max_model_len=proposed_config_dict.get(
                    "max_model_len", current_config.max_model_len
                ),
                gpu_memory_utilization=proposed_config_dict.get(
                    "gpu_memory_utilization", current_config.gpu_memory_utilization
                ),
                quantization=proposed_config_dict.get(
                    "quantization", current_config.quantization
                ),
                kv_cache_dtype=proposed_config_dict.get(
                    "kv_cache_dtype", current_config.kv_cache_dtype
                ),
                max_num_seqs=proposed_config_dict.get(
                    "max_num_seqs", current_config.max_num_seqs
                ),
                enable_prefix_caching=proposed_config_dict.get(
                    "enable_prefix_caching", current_config.enable_prefix_caching
                ),
                enforce_eager=proposed_config_dict.get(
                    "enforce_eager", current_config.enforce_eager
                ),
            )
        except ValueError as exc:
            logger.warning("oom_proposed_config_invalid", error=str(exc), event_id=event_id)
            try:
                await self._kg.update_oom_outcome(event_id=event_id, succeeded=False, resolution_config_id=None)
            except Exception as outcome_exc:
                logger.error("update_oom_outcome_failed", error=str(outcome_exc))
            return (False, None)

        # ------------------------------------------------------------------
        # STEP 6: persist the new configuration
        # ------------------------------------------------------------------
        config_id: str = await self._kg.write_configuration(new_config)

        logger.info(
            "oom_recovery.config_written",
            config_id=config_id,
            event_id=event_id,
            instance_id=instance_id,
        )

        # ------------------------------------------------------------------
        # STEPS 7-8: destructive actions — destroy then provision
        # STEP 9 (update_oom_outcome) is guaranteed by try/finally
        # ------------------------------------------------------------------
        succeeded = False
        try:
            await destroy_fn(instance_id)

            new_instance_id = await provision_fn(new_config)
            succeeded = new_instance_id is not None

            logger.info(
                "oom_recovery.provision_result",
                succeeded=succeeded,
                new_instance_id=new_instance_id,
                event_id=event_id,
            )
        finally:
            try:
                await self._kg.update_oom_outcome(
                    event_id=event_id,
                    succeeded=succeeded,
                    resolution_config_id=config_id if succeeded else None,
                )
            except Exception as outcome_exc:
                logger.error(
                    "update_oom_outcome_failed",
                    event_id=event_id,
                    error=str(outcome_exc),
                )

        return (succeeded, new_config if succeeded else None)
