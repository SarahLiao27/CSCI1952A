import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def generate_rewards(base_url, model, messages, temperature, sample_count, max_tokens):
    """Request reward candidates from a local chat-completions server."""
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    responses = []
    usage = dict.fromkeys(("prompt_tokens", "completion_tokens", "total_tokens"), 0)

    while len(responses) < sample_count:
        count = 1
        request = Request(
            endpoint,
            data=json.dumps({
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "n": count,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=300) as reply:
                result = json.load(reply)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Model server at {endpoint} returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"Request to {model} at {endpoint} failed: {exc}") from exc

        choices = result["choices"]
        if len(choices) != count or any(not choice["message"].get("content") for choice in choices):
            raise RuntimeError(f"Expected {count} nonempty completions from {model} at {endpoint}")
        responses.extend(choices)
        for key in usage:
            usage[key] += (result.get("usage") or {}).get(key, 0)

    return responses, usage
