"""Call the course Qwen service using only the Python standard library."""
import argparse
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def normalize_base_url(base_url):
    """Accept one service URL, including a file ending in CRLF or whitespace."""
    url = (base_url or "").strip().rstrip("/")
    try:
        parts = urlsplit(url)
        port = parts.port
        valid = (parts.scheme in ("http", "https") and parts.hostname
                 and parts.hostname.upper() != "NODE"
                 and parts.path == "/v1" and not parts.query
                 and not parts.fragment and parts.username is None
                 and parts.password is None and port != 0
                 and not any(c.isspace() for c in url))
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Set server-url.txt to this run's SERVICE_BASE_URL "
                         "(for example http://compute-node:8000/v1). "
                         "Replace NODE and keep the /v1 suffix.")
    return url


def chat(messages, base_url, max_tokens=256, temperature=0.7):
    """Return the full JSON response, including choices and token usage."""
    base_url = normalize_base_url(base_url)
    payload = {
        "model": "qwen3.8-27b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = Request(
        base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=300) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(
            f"Qwen service returned HTTP {exc.code}. Check the current service "
            "log, /v1 address, request length and model name qwen3.8-27b."
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            "Cannot reach the Qwen service or the request timed out. "
            "Wait for 'Application startup complete', confirm the service job "
            "is still running, and update server-url.txt after each restart."
        ) from exc
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError("Qwen service returned invalid JSON; check its log and URL.") from exc
    try:
        content = result["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("content is not text")
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Qwen response has no text answer; check the service log.") from exc
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("QWEN_BASE_URL"))
    parser.add_argument("--check-url", action="store_true",
                        help="validate and print the URL without a network request")
    parser.add_argument("--prompt", default="37加58等于多少？只输出数字。")
    args = parser.parse_args()
    try:
        url = normalize_base_url(args.base_url)
        if args.check_url:
            print(url)
        else:
            result = chat([{"role": "user", "content": args.prompt}], url,
                          max_tokens=64, temperature=0)
            print(result["choices"][0]["message"]["content"])
            print(json.dumps(result.get("usage", {}), ensure_ascii=False))
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
