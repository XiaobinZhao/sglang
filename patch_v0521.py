#!/usr/bin/env python3
"""Semantic port of v0.5.20_eagle_dsa_hang_fix onto sglang v0.5.21.

Same fix set as patch_v0520.py (v0.5.19 fix3 semantics). 34 of 37 patches
carry over verbatim; three were re-anchored for the v0.5.21 refactor:
  - P5b : base_runner.py dummy-run PP1 slice now keyed off get_parallel().pp_rank
          and mr.attn_cp_size (v0.5.21 renamed mr.ps.* accesses)
  - P22 : mooncake/conn.py helpers re-anchored to the daemon-thread bootstrap
          launcher (v0.5.21 replaced the inline Thread().start() block)
  - P23 : bootstrap loop header now `while not self._stopped` + recv() factory

Fix items:
 1. dsa_backend.py  : wrap 4 q8kv8 buffer allocations in torch.inference_mode(False)
 2. metadata-None guards at CP access points (dsa/utils.py cal_padded_tokens,
    dsa_indexer.py CP gathers, index_topk_share.py gather) + base_runner.py
    dummy-run PP1 keeps full PP tensors (match PP0's no-split dummy send).
    is_cp_active itself stays stock (PP0 send-side split depends on it).
 3. scheduler.py    : _abort_disagg_request_before_queue (fast-fail aborted PD reqs)
 4. mooncake/conn.py: orphan failed room TTL cleanup + late-bootstrap Failed push
                      (adapted to v0.5.21 conclude_failure API) + record_failure override
 5. common/conn.py  : KVTransferError.status_code + failure_status_codes propagation
 6. decode.py       : _resolve_kv_failure (exact HTTP status propagation)
"""
import sys
from pathlib import Path

BASE = Path(__file__).parent / "python" / "sglang" / "srt"

# (relpath, tag, old, new, expected_count)
PATCHES = []


def add(rel, tag, old, new, count=1):
    PATCHES.append((rel, tag, old, new, count))


# ---------------- 1. layers/attention/dsa_backend.py ----------------
REL_DSA = "layers/attention/dsa_backend.py"

add(REL_DSA, "P1 born_q_buf",
    """            buf = torch.zeros(
                (num_tokens, padded_heads, head_dim),
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            self._q8kv8_born_q_buf = buf""",
    """            with torch.inference_mode(False):
                buf = torch.zeros(
                    (num_tokens, padded_heads, head_dim),
                    dtype=torch.float8_e4m3fn,
                    device=device,
                )
            self._q8kv8_born_q_buf = buf""")

add(REL_DSA, "P2 qpad_buf",
    """                buf = torch.zeros(
                    (num_tokens, required_padding, head_dim),
                    dtype=torch.float8_e4m3fn,
                    device=dev,
                )
                self._q8kv8_qpad_buf = buf""",
    """                with torch.inference_mode(False):
                    buf = torch.zeros(
                        (num_tokens, required_padding, head_dim),
                        dtype=torch.float8_e4m3fn,
                        device=dev,
                    )
                self._q8kv8_qpad_buf = buf""")

add(REL_DSA, "P3 kv_buf",
    """            kv_buf = torch.empty(
                (total_kv_rows, head_dim),
                dtype=torch.float8_e4m3fn,
                device=dev,
            )
            self._q8kv8_kv_buf = kv_buf""",
    """            with torch.inference_mode(False):
                kv_buf = torch.empty(
                    (total_kv_rows, head_dim),
                    dtype=torch.float8_e4m3fn,
                    device=dev,
                )
            self._q8kv8_kv_buf = kv_buf""")

add(REL_DSA, "P4 out_bufs",
    """            out_bufs = (
                torch.empty(
                    s_q, pad_heads, v_head_dim, dtype=torch.bfloat16, device=dev
                ),
                torch.empty(s_q, pad_heads, dtype=torch.float32, device=dev),
                torch.empty(s_q, pad_heads, dtype=torch.float32, device=dev),
            )
            self._q8kv8_out_bufs = out_bufs""",
    """            with torch.inference_mode(False):
                out_bufs = (
                    torch.empty(
                        s_q, pad_heads, v_head_dim, dtype=torch.bfloat16, device=dev
                    ),
                    torch.empty(s_q, pad_heads, dtype=torch.float32, device=dev),
                    torch.empty(s_q, pad_heads, dtype=torch.float32, device=dev),
                )
            self._q8kv8_out_bufs = out_bufs""")

# ---------------- 2b. layers/cp/utils.py (restored) + runner/base_runner.py ----------------
REL_CP = "layers/cp/utils.py"

# ---------------- 2b. metadata None guards at CP access points ----------------
# is_cp_active itself stays upstream-stock: dummy runs legitimately reach
# cal_padded_tokens / indexer gathers with attn_cp_metadata unbuilt, and the
# PP0 cp-split send-side decision relies on stock can_apply semantics. Guard
# only the direct metadata accesses instead.
REL_DSAU = "layers/attention/dsa/utils.py"

add(REL_DSAU, "P5c cal_padded_tokens metadata None guard",
    """    # CP-v2 already pads each rank-local shard to its physical size
    if is_cp_active(forward_batch):
        return forward_batch.attn_cp_metadata.per_rank_actual_token[
            get_parallel().attn_cp_rank
        ]""",
    """    # CP-v2 already pads each rank-local shard to its physical size
    # (0520fix) Dummy/warmup runs enter with attn_cp_metadata unbuilt; fall
    # through to the non-CP padding path instead of crashing on None.
    if is_cp_active(forward_batch) and forward_batch.attn_cp_metadata is not None:
        return forward_batch.attn_cp_metadata.per_rank_actual_token[
            get_parallel().attn_cp_rank
        ]""")

REL_IDX = "layers/attention/dsa/dsa_indexer.py"

add(REL_IDX, "P5d-1 indexer alt-stream CP gather guard",
    """        elif self.alt_stream is not None and is_cp_active(forward_batch):""",
    """        elif (
            self.alt_stream is not None
            and is_cp_active(forward_batch)
            and forward_batch.attn_cp_metadata is not None
        ):""")

add(REL_IDX, "P5d-2 indexer CP gather guard",
    """        # allgather+rerrange
        if is_cp_active(forward_batch):
            key = get_cp_strategy().materialize_full_indexer_k_cache(key, forward_batch)
        return query, key, weights_raw""",
    """        # allgather+rerrange
        # (0520fix) Skip the CP gather for dummy/warmup batches whose
        # attn_cp_metadata is unbuilt; the gather requires it.
        if (
            is_cp_active(forward_batch)
            and forward_batch.attn_cp_metadata is not None
        ):
            key = get_cp_strategy().materialize_full_indexer_k_cache(key, forward_batch)
        return query, key, weights_raw""")

REL_ITS = "layers/attention/index_topk_share.py"

add(REL_ITS, "P5e topk share gather guard",
    """        if (
            topk_indices is not None
            and self.should_publish
            and is_cp_active(self._forward_batch)
        ):""",
    """        if (
            topk_indices is not None
            and self.should_publish
            and is_cp_active(self._forward_batch)
            and self._forward_batch.attn_cp_metadata is not None
        ):""")

REL_BR = "model_executor/runner/base_runner.py"

add(REL_BR, "P5b dummy-run PP1 keeps full PP tensors (match PP0 no-split)",
    """        if get_parallel().pp_size > 1:
            # PP0 already cp-split hidden_states before send.
            pp_hidden_tokens = num_tokens
            if (
                capture_forward_mode == ForwardMode.EXTEND
                and get_parallel().pp_rank != 0
                and mr.attn_cp_size > 1
            ):
                pp_hidden_tokens = num_tokens // mr.attn_cp_size""",
    """        if get_parallel().pp_size > 1:
            # PP0 already cp-split hidden_states before send.
            pp_hidden_tokens = num_tokens
            # (0521fix) Dummy runs never build attn_cp_metadata, so PP0's model
            # forward (dsa_use_prefill_cp requires non-None metadata) never
            # CP-splits the hidden states it sends. Drop the unconditional
            # num_tokens // attn_cp_size slice on PP ranks != 0: it mismatched
            # PP0's unsliced send and crashed rotary_emb (positions 8 vs k 1).""")

# ---------------- 3. managers/scheduler.py ----------------
REL_SCHED = "managers/scheduler.py"

add(REL_SCHED, "P6 import is_aborted",
    """from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_dsa_seed_metadata_dim,
    prepare_abort,
    unified_memory_disagg_move_gate,
)""",
    """from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_dsa_seed_metadata_dim,
    is_aborted,
    prepare_abort,
    unified_memory_disagg_move_gate,
)""")

add(REL_SCHED, "P7 PREFILL pre-queue abort",
    """        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._prefetch_kvcache(req)
            self.disagg_prefill_bootstrap_queue.add(
                req, self.model_config.num_key_value_heads
            )""",
    """        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            if self._abort_disagg_request_before_queue(req):
                return
            self._prefetch_kvcache(req)
            self.disagg_prefill_bootstrap_queue.add(
                req, self.model_config.num_key_value_heads
            )""")

add(REL_SCHED, "P8 DECODE pre-queue abort",
    """        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.disagg_decode_prealloc_queue.add(req, is_retracted=is_retracted)""",
    """        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            if not is_retracted and self._abort_disagg_request_before_queue(req):
                return
            self.disagg_decode_prealloc_queue.add(req, is_retracted=is_retracted)""")

add(REL_SCHED, "P9 _abort_disagg_request_before_queue",
    """        else:
            raise ValueError(f"Invalid {self.disaggregation_mode=}")

    def _reject_sampling_mask_request(self, req: Req, error_msg: str) -> None:""",
    """        else:
            raise ValueError(f"Invalid {self.disaggregation_mode=}")

    def _abort_disagg_request_before_queue(self, req: Req) -> bool:
        \"\"\"Finish an already-aborted PD request before it enters a KV queue.\"\"\"
        if not is_aborted(req):
            return False

        has_bootstrap = (
            req.bootstrap_room is not None and req.bootstrap_host is not None
        )
        if (
            self.disaggregation_mode == DisaggregationMode.PREFILL
            and req.disagg_kv_sender is None
            and has_bootstrap
        ):
            if not self.disagg_prefill_bootstrap_queue.create_sender(
                req, self.model_config.num_key_value_heads
            ):
                return True

        abort_reason = req.to_finish or req.finished_reason
        sender = req.disagg_kv_sender
        if sender is not None and hasattr(sender, "abort"):
            kv_mgr = getattr(sender, "kv_mgr", None)
            record_failure = getattr(kv_mgr, "record_failure", None)
            if isinstance(abort_reason, FINISH_ABORT) and record_failure is not None:
                record_failure(
                    req.bootstrap_room,
                    abort_reason.message,
                    abort_reason.status_code,
                )
            sender.abort()
            notify_decode = getattr(kv_mgr, "try_notify_decode_failure_and_clear", None)
            if notify_decode is not None:
                try:
                    notify_decode(req.bootstrap_room)
                except Exception as error:
                    logger.warning(
                        "Failed to notify decode about pre-queue abort for room %s: %s",
                        req.bootstrap_room,
                        error,
                    )

        if req.finished_reason is None and req.to_finish is not None:
            req.finished_reason = req.to_finish
            req.to_finish = None
        req.time_stats.set_completion_time()
        self.output_streamer.stream_output([req], req.return_logprob)
        return True

    def _reject_sampling_mask_request(self, req: Req, error_msg: str) -> None:""")

# ---------------- 4. disaggregation/common/conn.py ----------------
REL_COMMON = "disaggregation/common/conn.py"

add(REL_COMMON, "P10 KVTransferError.status_code",
    """class KVTransferError(Exception):
    def __init__(
        self,
        bootstrap_room: int,
        failure_reason: str,
        is_from_another_rank: bool = False,
    ):
        super().__init__(failure_reason)
        self.bootstrap_room = bootstrap_room
        self.failure_reason = failure_reason
        self.is_from_another_rank = is_from_another_rank""",
    """class KVTransferError(Exception):
    def __init__(
        self,
        bootstrap_room: int,
        failure_reason: str,
        is_from_another_rank: bool = False,
        status_code: Optional[int] = None,
    ):
        super().__init__(failure_reason)
        self.bootstrap_room = bootstrap_room
        self.failure_reason = failure_reason
        self.is_from_another_rank = is_from_another_rank
        self.status_code = status_code""")

add(REL_COMMON, "P11 failure_status_codes table",
    """        self.failure_records: Dict[int, str] = {}
        self.failure_lock = threading.Lock()""",
    """        self.failure_records: Dict[int, str] = {}
        self.failure_status_codes: Dict[int, int] = {}
        self.failure_lock = threading.Lock()""")

add(REL_COMMON, "P12 record_failure status_code",
    """    def record_failure(self, bootstrap_room: int, failure_reason: str):
        with self.failure_lock:
            self.failure_records[bootstrap_room] = failure_reason""",
    """    def record_failure(
        self,
        bootstrap_room: int,
        failure_reason: str,
        status_code: Optional[int] = None,
    ):
        with self.failure_lock:
            self.failure_records[bootstrap_room] = failure_reason
            if status_code is not None:
                self.failure_status_codes[bootstrap_room] = int(status_code)""")

add(REL_COMMON, "P13 _encode_kv_status_message status_code frame",
    """    def _encode_kv_status_message(
        self,
        *,
        bootstrap_room: int,
        status: KVPoll,
        failure_reason: Optional[str],
    ) -> List[bytes]:
        parts = [
            str(bootstrap_room).encode("ascii"),
            str(int(status)).encode("ascii"),
            str(self._prefill_unique_rank()).encode("ascii"),
        ]
        if self.kv_status_msg_carries_reason:
            parts.append((failure_reason or "").encode("utf-8"))
        if self.kv_status_msg_tag is not None:
            parts.insert(0, self.kv_status_msg_tag)
        return parts""",
    """    def _encode_kv_status_message(
        self,
        *,
        bootstrap_room: int,
        status: KVPoll,
        failure_reason: Optional[str],
        status_code: Optional[int] = None,
    ) -> List[bytes]:
        parts = [
            str(bootstrap_room).encode("ascii"),
            str(int(status)).encode("ascii"),
            str(self._prefill_unique_rank()).encode("ascii"),
        ]
        if self.kv_status_msg_carries_reason:
            parts.append((failure_reason or "").encode("utf-8"))
            if status_code is not None:
                parts.append(str(int(status_code)).encode("ascii"))
        if self.kv_status_msg_tag is not None:
            parts.insert(0, self.kv_status_msg_tag)
        return parts""")

add(REL_COMMON, "P14 parse_kv_status_message_with_code",
    """        return bootstrap_room, status, prefill_rank, failure_reason

    def send_kv_status_message(""",
    """        return bootstrap_room, status, prefill_rank, failure_reason

    def parse_kv_status_message_with_code(
        self, msg: List[bytes]
    ) -> Optional[Tuple[int, int, int, Optional[str], Optional[int]]]:
        \"\"\"Decode a prefill status message plus its optional status code frame.\"\"\"
        parsed = self.parse_kv_status_message(msg)
        if parsed is None:
            return None
        if self.kv_status_msg_tag is not None:
            if not msg or msg[0] != self.kv_status_msg_tag:
                return None
            msg = msg[1:]
        status_code = None
        if self.kv_status_msg_carries_reason and len(msg) > 4 and msg[4]:
            try:
                status_code = int(msg[4].decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                logger.warning("Dropping unparsable prefill status code frame")
                status_code = None
        return (*parsed, status_code)

    def send_kv_status_message(""")

add(REL_COMMON, "P15 send_kv_status_message status_code",
    """    def send_kv_status_message(
        self,
        *,
        targets: List[Tuple[str, int]],
        bootstrap_room: int,
        status: KVPoll,
        failure_reason: Optional[str] = None,
    ) -> None:
        \"\"\"Push of a terminal transfer status to decode endpoints.\"\"\"
        if not targets:
            return
        parts = self._encode_kv_status_message(
            bootstrap_room=bootstrap_room,
            status=status,
            failure_reason=failure_reason,
        )""",
    """    def send_kv_status_message(
        self,
        *,
        targets: List[Tuple[str, int]],
        bootstrap_room: int,
        status: KVPoll,
        failure_reason: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> None:
        \"\"\"Push of a terminal transfer status to decode endpoints.\"\"\"
        if not targets:
            return
        parts = self._encode_kv_status_message(
            bootstrap_room=bootstrap_room,
            status=status,
            failure_reason=failure_reason,
            status_code=status_code,
        )""")

add(REL_COMMON, "P16a conclude_transfer signature",
    """    def conclude_transfer(
        self,
        *,
        bootstrap_room: int,
        status: KVPoll,
        targets: Optional[List[Tuple[str, int]]] = None,
        failure_reason: Optional[str] = None,
    ) -> Optional[KVPoll]:""",
    """    def conclude_transfer(
        self,
        *,
        bootstrap_room: int,
        status: KVPoll,
        targets: Optional[List[Tuple[str, int]]] = None,
        failure_reason: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> Optional[KVPoll]:""")

add(REL_COMMON, "P16b conclude_transfer send call",
    """        self.update_status(bootstrap_room, status)
        self.send_kv_status_message(
            targets=targets,
            bootstrap_room=bootstrap_room,
            status=status,
            failure_reason=failure_reason,
        )
        return status""",
    """        self.update_status(bootstrap_room, status)
        self.send_kv_status_message(
            targets=targets,
            bootstrap_room=bootstrap_room,
            status=status,
            failure_reason=failure_reason,
            status_code=status_code,
        )
        return status""")

add(REL_COMMON, "P16c conclude_failure status_code",
    """    def conclude_failure(
        self,
        *,
        bootstrap_room: int,
        failure_reason: str,
        targets: Optional[List[Tuple[str, int]]] = None,
    ) -> Optional[KVPoll]:
        \"\"\"Record the reason, mark the room Failed and tell decode.\"\"\"
        return self.conclude_transfer(
            bootstrap_room=bootstrap_room,
            status=KVPoll.Failed,
            targets=targets,
            failure_reason=failure_reason,
        )""",
    """    def conclude_failure(
        self,
        *,
        bootstrap_room: int,
        failure_reason: str,
        targets: Optional[List[Tuple[str, int]]] = None,
        status_code: Optional[int] = None,
    ) -> Optional[KVPoll]:
        \"\"\"Record the reason, mark the room Failed and tell decode.\"\"\"
        return self.conclude_transfer(
            bootstrap_room=bootstrap_room,
            status=KVPoll.Failed,
            targets=targets,
            failure_reason=failure_reason,
            status_code=status_code,
        )""")

add(REL_COMMON, "P17a apply_prefill_status signature",
    """    def apply_prefill_status(
        self,
        *,
        bootstrap_room: int,
        status: int,
        prefill_rank: int,
        failure_reason: Optional[str] = None,
    ) -> None:""",
    """    def apply_prefill_status(
        self,
        *,
        bootstrap_room: int,
        status: int,
        prefill_rank: int,
        failure_reason: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> None:""")

add(REL_COMMON, "P17b apply_prefill_status Failed branch",
    """        if status == KVPoll.Failed:
            self.record_failure(
                bootstrap_room, failure_reason or self.DEFAULT_PREFILL_FAILURE_REASON
            )
            self.update_status(bootstrap_room, KVPoll.Failed)
            return""",
    """        if status == KVPoll.Failed:
            self.record_failure(
                bootstrap_room,
                failure_reason or self.DEFAULT_PREFILL_FAILURE_REASON,
                status_code,
            )
            self.update_status(bootstrap_room, KVPoll.Failed)
            return""")

add(REL_COMMON, "P18 CommonKVSender.clear cleanup",
    """    def clear(self) -> None:
        self.kv_mgr.request_status.pop(self.bootstrap_room, None)
        if hasattr(self.kv_mgr, "req_to_decode_prefix_len"):""",
    """    def clear(self) -> None:
        self.kv_mgr.request_status.pop(self.bootstrap_room, None)
        getattr(self.kv_mgr, "failure_status_codes", {}).pop(self.bootstrap_room, None)
        if hasattr(self.kv_mgr, "failure_timestamps"):
            self.kv_mgr.failure_timestamps.pop(self.bootstrap_room, None)
        if hasattr(self.kv_mgr, "required_dst_info_num_table"):
            self.kv_mgr.required_dst_info_num_table.pop(self.bootstrap_room, None)
        if hasattr(self.kv_mgr, "req_to_decode_prefix_len"):""")

add(REL_COMMON, "P19 abort keeps first failure reason (x2)",
    """    def abort(self):
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            "Aborted by AbortReq.",
        )
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)""",
    """    def abort(self):
        with self.kv_mgr.failure_lock:
            has_failure_reason = self.bootstrap_room in self.kv_mgr.failure_records
        if not has_failure_reason:
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                "Aborted by AbortReq.",
            )
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)""",
    count=2)

add(REL_COMMON, "P20 CommonKVReceiver.clear cleanup",
    """    def clear(self) -> None:
        self.kv_mgr.request_status.pop(self.bootstrap_room, None)
        self.kv_mgr.required_prefill_response_num_table.pop(self.bootstrap_room, None)
        self.kv_mgr.prefill_response_tracker.pop(self.bootstrap_room, None)
        self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].discard(
            self.bootstrap_room
        )""",
    """    def clear(self) -> None:
        self.kv_mgr.request_status.pop(self.bootstrap_room, None)
        self.kv_mgr.required_prefill_response_num_table.pop(self.bootstrap_room, None)
        self.kv_mgr.prefill_response_tracker.pop(self.bootstrap_room, None)
        getattr(self.kv_mgr, "failure_status_codes", {}).pop(self.bootstrap_room, None)
        if hasattr(self.kv_mgr, "failure_timestamps"):
            self.kv_mgr.failure_timestamps.pop(self.bootstrap_room, None)
        self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].discard(
            self.bootstrap_room
        )""")

# ---------------- 5. disaggregation/mooncake/conn.py ----------------
REL_MC = "disaggregation/mooncake/conn.py"

add(REL_MC, "P21 init orphan tables",
    """            self.session_failures = defaultdict(int)
            self.failed_sessions = set()
            self.session_lock = threading.Lock()
            self.start_prefill_thread()""",
    """            self.session_failures = defaultdict(int)
            self.failed_sessions = set()
            self.session_lock = threading.Lock()
            self.required_dst_info_num_table: dict[int, int] = {}
            self.failure_timestamps: dict[int, float] = {}
            self.orphan_failed_room_ttl = (
                envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.get()
            )
            self._orphan_failed_room_cleanup_interval = 60.0
            self._next_orphan_failed_room_cleanup_time = 0.0
            self.start_prefill_thread()""")

add(REL_MC, "P22 orphan/failure helpers",
    """        t = threading.Thread(target=bootstrap_thread, daemon=True)
        t.start()
        self._worker_threads.append(t)

    def start_decode_thread(self):""",
    """        t = threading.Thread(target=bootstrap_thread, daemon=True)
        t.start()
        self._worker_threads.append(t)

    def record_failure(
        self,
        bootstrap_room: int,
        failure_reason: str,
        status_code: Optional[int] = None,
    ):
        super().record_failure(bootstrap_room, failure_reason, status_code)
        if hasattr(self, "failure_timestamps"):
            with self.failure_lock:
                self.failure_timestamps.setdefault(bootstrap_room, time.monotonic())
            self.maybe_cleanup_orphan_failed_rooms()

    def _is_decode_metadata_ready(self, room: int) -> bool:
        required = self.required_dst_info_num_table.get(room)
        return (
            required is not None
            and room in self.transfer_infos
            and len(self.transfer_infos[room]) >= required
        )

    def try_notify_decode_failure_and_clear(self, room: int) -> bool:
        if self.request_status.get(room) != KVPoll.Failed:
            return False
        if not self._is_decode_metadata_ready(room):
            return False
        with self.failure_lock:
            failure_reason = self.failure_records.get(room)
            status_code = self.failure_status_codes.get(room)
        self.conclude_failure(
            bootstrap_room=room,
            failure_reason=failure_reason
            or "Failed to get kvcache from prefill instance, it might be dead",
            status_code=status_code,
        )
        self.clear_room(room)
        return True

    def clear_room(self, room: int) -> None:
        self.request_status.pop(room, None)
        with self.failure_lock:
            self.failure_records.pop(room, None)
            self.failure_status_codes.pop(room, None)
            self.failure_timestamps.pop(room, None)
        self.transfer_infos.pop(room, None)
        self.req_to_decode_prefix_len.pop(room, None)
        self.required_dst_info_num_table.pop(room, None)
        if hasattr(self, "_deferred_ack_targets"):
            self._deferred_ack_targets.pop(room, None)

    def maybe_cleanup_orphan_failed_rooms(self) -> None:
        now = time.monotonic()
        if now < self._next_orphan_failed_room_cleanup_time:
            return
        self._next_orphan_failed_room_cleanup_time = (
            now + self._orphan_failed_room_cleanup_interval
        )
        with self.failure_lock:
            failure_timestamps = list(self.failure_timestamps.items())
        for room, failed_at in failure_timestamps:
            if now - failed_at < self.orphan_failed_room_ttl:
                continue
            if self._is_decode_metadata_ready(room):
                self.try_notify_decode_failure_and_clear(room)
            else:
                logger.warning(
                    "Cleaning up orphan failed Mooncake room %s after %.1fs",
                    room,
                    now - failed_at,
                )
                self.clear_room(room)

    def start_decode_thread(self):""")

add(REL_MC, "P23 bootstrap loop cleanup hook",
    """            while not self._stopped:
                waiting_req_bytes = recv()
                if waiting_req_bytes is None:
                    continue
                room = waiting_req_bytes[0].decode("ascii")""",
    """            while not self._stopped:
                waiting_req_bytes = recv()
                if waiting_req_bytes is None:
                    continue
                self.maybe_cleanup_orphan_failed_rooms()
                room = waiting_req_bytes[0].decode("ascii")""")

add(REL_MC, "P24 late bootstrap Failed push",
    """                else:
                    required_dst_info_num = int(waiting_req_bytes[7].decode("ascii"))
                    room = int(room)
                    if room not in self.transfer_infos:
                        self.transfer_infos[room] = {}

                    self.transfer_infos[room][mooncake_session_id] = (
                        TransferInfo.from_zmq(waiting_req_bytes)
                    )
                    # NOTE: after bootstrapping we can mark the req as waiting for input
                    if len(self.transfer_infos[room]) == required_dst_info_num:
                        self.resolve_kv_replica_factor(self.transfer_infos[room])""",
    """                else:
                    required_dst_info_num = int(waiting_req_bytes[7].decode("ascii"))
                    room = int(room)
                    self.required_dst_info_num_table[room] = required_dst_info_num
                    if room not in self.transfer_infos:
                        self.transfer_infos[room] = {}

                    self.transfer_infos[room][mooncake_session_id] = (
                        TransferInfo.from_zmq(waiting_req_bytes)
                    )
                    # NOTE: after bootstrapping we can mark the req as waiting for input
                    if len(self.transfer_infos[room]) == required_dst_info_num:
                        if self.request_status.get(room) == KVPoll.Failed:
                            self.try_notify_decode_failure_and_clear(room)
                            continue
                        self.resolve_kv_replica_factor(self.transfer_infos[room])""")

add(REL_MC, "P25 decode thread 5-tuple + status_code",
    """                parsed = self.parse_kv_status_message(msg)
                if parsed is None:
                    continue
                room, status, prefill_rank, reason = parsed
                self.apply_prefill_status(
                    bootstrap_room=room,
                    status=status,
                    prefill_rank=prefill_rank,
                    failure_reason=reason,
                )""",
    """                parsed = self.parse_kv_status_message_with_code(msg)
                if parsed is None:
                    continue
                room, status, prefill_rank, reason, status_code = parsed
                self.apply_prefill_status(
                    bootstrap_room=room,
                    status=status,
                    prefill_rank=prefill_rank,
                    failure_reason=reason,
                    status_code=status_code,
                )""")

add(REL_MC, "P26 failure_exception carries status_code",
    """        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(self.bootstrap_room, None)
        is_propagated = failure_reason is None
        if is_propagated:
            failure_reason = "Failed due to an unknown reason from another rank"
        raise KVTransferError(
            self.bootstrap_room, failure_reason, is_from_another_rank=is_propagated
        )""",
    """        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(self.bootstrap_room, None)
            status_code = self.kv_mgr.failure_status_codes.pop(
                self.bootstrap_room, None
            )
        self.clear()
        is_propagated = failure_reason is None
        if is_propagated:
            failure_reason = "Failed due to an unknown reason from another rank"
        raise KVTransferError(
            self.bootstrap_room,
            failure_reason,
            is_from_another_rank=is_propagated,
            status_code=status_code,
        )""")

# ---------------- 6. disaggregation/decode.py ----------------
REL_DEC = "disaggregation/decode.py"

add(REL_DEC, "P27 _resolve_kv_failure helper",
    """def _bootstrap_addr(req: Req) -> str:
    # FIXME: make a property of a req
    return NetworkAddress(req.bootstrap_host, req.bootstrap_port).to_host_port_str()""",
    """def _bootstrap_addr(req: Req) -> str:
    # FIXME: make a property of a req
    return NetworkAddress(req.bootstrap_host, req.bootstrap_port).to_host_port_str()


def _resolve_kv_failure(
    kv_receiver: CommonKVReceiver, fallback_message: str
) -> Tuple[str, bool, int]:
    \"\"\"Return the message, propagation flag, and HTTP status for a KV failure.\"\"\"
    try:
        kv_receiver.failure_exception()
    except Exception as error:
        is_propagated = getattr(error, "is_from_another_rank", False)
        status_code = getattr(error, "status_code", None)
        if status_code is not None:
            return (
                getattr(error, "failure_reason", str(error)),
                is_propagated,
                status_code,
            )
        return (
            f"{fallback_message} with exception {error}",
            is_propagated,
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )
    return fallback_message, False, HTTPStatus.INTERNAL_SERVER_ERROR""")

add(REL_DEC, "P28 prealloc-queue failure resolution",
    """            elif poll == KVPoll.Failed:
                error_message = f"Decode handshake failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                is_propagated = False
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                    is_propagated = getattr(e, "is_from_another_rank", False)
                # Mute error message for propagated exceptions to avoid duplicate logging
                if is_propagated:
                    logger.debug(error_message)
                else:
                    logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )""",
    """            elif poll == KVPoll.Failed:
                fallback_message = f"Decode handshake failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                error_message, is_propagated, status_code = _resolve_kv_failure(
                    decode_req.kv_receiver, fallback_message
                )
                # Mute error message for propagated exceptions to avoid duplicate logging
                if is_propagated:
                    logger.debug(error_message)
                else:
                    logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=status_code,
                )""")

add(REL_DEC, "P29a transfer-queue failure resolution",
    """                is_propagated = False
                if poll == KVPoll.Failed:
                    try:
                        decode_req.kv_receiver.failure_exception()
                    except Exception as e:
                        error_message += f" with exception {e}"
                        is_propagated = getattr(e, "is_from_another_rank", False)""",
    """                is_propagated = False
                status_code = HTTPStatus.INTERNAL_SERVER_ERROR
                if poll == KVPoll.Failed:
                    error_message, is_propagated, status_code = _resolve_kv_failure(
                        decode_req.kv_receiver, error_message
                    )""")

add(REL_DEC, "P29b transfer-queue prepare_abort status_code",
    """                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                self.scheduler.output_streamer.stream_output(""",
    """                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=status_code,
                )
                self.scheduler.output_streamer.stream_output(""")


# ---------------- driver ----------------
def main():
    dry = "--check" in sys.argv
    contents = {}
    errors = []

    for rel, tag, old, new, count in PATCHES:
        p = BASE / rel
        if rel not in contents:
            contents[rel] = p.read_text()
        s = contents[rel]
        n = s.count(old)
        if n != count:
            errors.append(f"[{tag}] {rel}: old appears {n}x (expect {count})")
        elif not dry:
            contents[rel] = s.replace(old, new)

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print("  " + e)
        sys.exit(1)

    if dry:
        print(f"DRY-RUN OK: all {len(PATCHES)} patches validated")
        return

    for rel, s in contents.items():
        (BASE / rel).write_text(s)
    print(f"APPLIED {len(PATCHES)} patches to {len(contents)} files:")
    for rel in contents:
        print("  " + rel)


if __name__ == "__main__":
    main()
