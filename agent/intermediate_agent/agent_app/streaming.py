"""Display text as it arrives and report timing for the single-call lesson."""

from time import perf_counter


def stream_answer(agent, question):
    started = perf_counter()
    first_text_seconds = None
    final_state = None
    has_text = False
    # Token events display the answer; state events retain final message metadata.
    for mode, event in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode=["messages", "values"],
    ):
        if mode == "messages":
            chunk, _metadata = event
            text = chunk.text
            if text:
                if first_text_seconds is None:
                    first_text_seconds = perf_counter() - started
                has_text = True
                print(text, end="", flush=True)
        elif mode == "values":
            final_state = event
    elapsed = perf_counter() - started
    print(flush=True)
    if not has_text or not final_state or not final_state.get("messages"):
        raise ValueError("The model returned no streamed answer text.")
    return final_state, first_text_seconds, elapsed


def print_timings(response, first_text_seconds, elapsed):
    """Ollama durations are nanoseconds. Missing metrics are not shown as zero."""
    print(f"\nTiming: first text {first_text_seconds:.2f}s | agent call {elapsed:.2f}s")
    metadata = response.response_metadata
    for key, label in [("load_duration", "Model loading"),
                       ("prompt_eval_duration", "Prompt processing"),
                       ("eval_duration", "Text generation")]:
        value = metadata.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            print(f"  {label}: {value / 1e9:.2f}s")
    count = metadata.get("eval_count")
    duration = metadata.get("eval_duration")
    if isinstance(count, (int, float)) and isinstance(duration, (int, float)) and duration > 0:
        print(f"  Generation speed: {count / (duration / 1e9):.1f} tokens/s ({count} tokens)")
