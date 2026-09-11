"""vLLM 0.23.0 synchronous V1 bridge, no persistent GPU hidden-state tensors."""


def contexts(runner, scheduler_output, hidden_states):
    batch = runner.input_batch
    positions_cpu = runner.positions.cpu()
    result, offset = [], 0
    for row, request_id in enumerate(batch.req_ids):
        prompt_len = int(batch.num_prompt_tokens[row])
        computed = int(batch.num_computed_tokens_cpu[row])
        scheduled = scheduler_output.num_scheduled_tokens[request_id]
        context = {"request_id": request_id, "prompt_len": prompt_len, "hidden": None}
        # The last INPUT position, not the last draft/accepted output position.
        local = prompt_len - 1 - computed
        if 0 <= local < scheduled:
            position = offset + local
            if int(positions_cpu[position]) != prompt_len - 1:
                raise RuntimeError("Hidden-state position mismatch; refusing wrong retrieval features")
            context["hidden"] = hidden_states[position].detach().float().cpu().tolist()
        result.append(context)
        offset += scheduled
    return result


def finish_requests(runner, request_ids):
    drafter = getattr(runner, "drafter", None)
    engine = getattr(drafter, "engine", None)
    if engine is not None:
        for request_id in request_ids:
            engine.finish(request_id)
