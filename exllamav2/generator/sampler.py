    # Sampling

    output_tokens = torch.empty((batch_size, 1), dtype=torch.long)
    output_probs = torch.empty((batch_size, 1), dtype=torch.float)
    
    if return_top_tokens == 0:
        output_ktokens = none_tensor
        output_kprobs = none_tensor
    else:
        output_ktokens = torch.empty((batch_size, 1, return_top_tokens), dtype=torch.long)
        output_kprobs = torch.empty((batch_size, 1, return_top_tokens), dtype=torch.float)

    # Apply skew transformation before sampling
    if settings.skew != 0:
        # Adjust logits by skew. Skew > 0 accentuates larger logits, skew < 0 flattens the distribution
        skew_factor = torch.sign(logits) * (logits.abs() ** (1 + settings.skew))
        logits = skew_factor

    m = ext_c.sample_basic(
        logits,
        1.0 if settings.temperature_last else settings.temperature,
        settings.top_k,
        settings.top_p,
        settings.top_a,
        settings.min_p,
        settings.tfs,
        settings.typical,
        random,
        output_tokens,
        output_probs,
        output_kprobs,
        output_ktokens,
        logit_filter if logit_filter is not None else none_tensor,
        settings.mirostat,
        settings.mirostat_mu if settings.mirostat else [],
        settings.mirostat_tau,
        settings.mirostat_eta,
        settings.temperature if settings.temperature_last else 1.0,
        settings.min_temp,
        settings.max_temp,
        settings.temp_exponent,
        settings.smoothing_factor,
        settings.skew
    )
