"""Talk to the models: a self-hosted vLLM server, or a frontier API.

Every open-weight number in the paper came through :class:`VLLMClient`. The
seven models are served one at a time behind an OpenAI-compatible endpoint, so
the client is thin; what it is not thin about is the two things that decide
whether a run finishes and whether its output is scoreable.

Context overflow
----------------

ICL-FS puts about 30K tokens of demonstrations (HI-Small) or 22K (LI-Small) in
front of a case that can itself run to 70K, and the reasoning budget is another
32K. On the 128K models that overflows, and vLLM answers with one of two
``BadRequestError`` shapes rather than a truncated completion. Skipping those
cases would silently drop the largest context graphs from the evaluation, which
are exactly the ones the paper is about, so the client handles both shapes and
retries:

* ``max_tokens must be at least 1, got -N`` -- the prompt alone exceeds the
  window. Cut ``N`` tokens' worth of characters out of the middle of the user
  message and retry with a small output budget.
* ``maximum context length is X ... resulted in Y`` -- the prompt fits but
  prompt plus output does not. Clamp the output budget to what is left; if
  even that is negative, the prompt did not fit either, so truncate.

Truncation hits 124 of 3,753 HI-Small cases (3.3%) and 26 of 2,268 LI-Small
cases (1.1%) on the three 128K models under ICL-FS. The 256K models are
unaffected, and ICL-ZS fits everywhere.

Reasoning extraction
--------------------

Thinking models put their chain of thought somewhere other than the answer, and
where depends on the vLLM version and on whether the server was started with a
reasoning parser. All three routes are handled, because the traces are the
input to the four-step rubric and a missing one would silently become an empty
trace.

Credentials
-----------

Read from the process environment, never from a file this module goes looking
for. The pre-release client walked up the directory tree for a ``.env`` and
loaded the first one it found, so which account got billed depended on the
working directory.

Models that appear in no table are gone: the registry is built from
:data:`amlc.config.LLM_MODELS`, which raises here if the two drift apart.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, replace
from typing import Optional

from .. import config

logger = logging.getLogger(__name__)

#: Output budget for a retry after a context overflow. Generous: a verdict runs
#: to about 500 tokens once the thinking is done.
RETRY_MAX_TOKENS = 4096

#: Characters per token when converting a token overflow into a cut length,
#: with 30% of slack because the estimate is crude and a second overflow costs
#: another round trip.
CHARS_PER_TOKEN = 4
CUT_SLACK = 1.3

DEFAULT_VLLM_BASE_URL = config.VLLM_BASE_URL


@dataclass
class LLMResponse:
    """One completion, with the accounting the runner records."""

    content: str
    model: str
    usage: dict
    latency_ms: float
    raw_response: Optional[dict] = None


@dataclass(frozen=True)
class ModelConfig:
    """Serving and sampling parameters for one model."""

    name: str
    model_id: str          # must match --served-model-name on the vLLM server
    api_type: str = "vllm"
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1        # -1 disables it, which is the vLLM default
    min_p: float = 0.0     # 0.0 disables it
    max_tokens: int = config.LLM_MAX_TOKENS
    # Sent to the chat template on every call, as every reported run did.
    # Templates that do not read it ignore it; the ones that do are the Qwen3.5
    # pair, where it is what keeps thinking mode on.
    enable_thinking: bool = True
    # Optional vLLM sampling seed. None preserves unseeded direct calls.
    seed: Optional[int] = None


#: Officially recommended sampling parameters, per model card, thinking mode.
#: Every model is run in the mode its authors intended rather than under one
#: house setting, which is why these differ.
#:
#:   GPT-OSS                huggingface.co/openai/gpt-oss-120b/discussions/21
#:   Qwen3.5 (MoE, 27B)     model card, thinking mode
#:   Nemotron-3 Super/Nano  model card
_SAMPLING = config.LLM_SAMPLING

#: The evaluated models. Keyed off config.LLM_MODELS on purpose: a model added
#: to the paper's list without sampling parameters raises a KeyError here rather
#: than being served under a silent default.
MODELS = {
    name: ModelConfig(name=name, model_id=name, **_SAMPLING[name])
    for name in config.LLM_MODELS
}


class BaseClient:
    """Common surface: build a :class:`ModelConfig`, then :meth:`call`."""

    def __init__(self, model_config: ModelConfig):
        self.config = model_config

    def call(self, prompt: str, system_prompt: Optional[str] = None,
             max_tokens: Optional[int] = None) -> LLMResponse:
        """Send one prompt.

        ``max_tokens`` overrides the configured output budget for this call.
        """
        raise NotImplementedError


class VLLMClient(BaseClient):
    """A model served by vLLM behind its OpenAI-compatible API.

    Start the server with the model's own reasoning parser where it has one::

        python -m vllm.entrypoints.openai.api_server \\
            --model <hf id> --served-model-name <name in MODELS> \\
            --host 0.0.0.0 --port 18809 --reasoning-parser qwen3
    """

    def __init__(self, model_config: ModelConfig,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        super().__init__(model_config)
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL",
                                                   DEFAULT_VLLM_BASE_URL)
        # vLLM ignores the key unless it was started with one.
        self.api_key = api_key or config.VLLM_API_KEY

    @staticmethod
    def _truncate_user_message(messages: list, chars_to_cut: int) -> list:
        """Cut the middle out of the user message to fit the context window.

        The head carries the instruction preamble and the demonstrations and the
        tail carries the answer format, so both have to survive; what goes is
        the middle, which is transaction lines. The split is 85% head and the
        remainder tail, with at least 500 characters of tail.
        """
        msgs = [m.copy() for m in messages]
        for m in msgs:
            if m["role"] == "user":
                content = m["content"]
                if chars_to_cut >= len(content):
                    # Nothing sensible left to keep: hold on to the opening
                    # instructions and the answer format and drop the rest.
                    m["content"] = (
                        content[:2000]
                        + "\n\n[... content truncated to fit context window ...]\n\n"
                        + content[-1000:]
                    )
                else:
                    keep_head = int((len(content) - chars_to_cut) * 0.85)
                    keep_tail = len(content) - chars_to_cut - keep_head
                    keep_tail = max(keep_tail, 500)
                    m["content"] = (
                        content[:keep_head]
                        + "\n\n[... content truncated to fit context window ...]\n\n"
                        + content[-keep_tail:]
                    )
                break
        return msgs

    @staticmethod
    def _chars_to_cut(overflow_tokens: int) -> int:
        return int((overflow_tokens + RETRY_MAX_TOKENS) * CHARS_PER_TOKEN * CUT_SLACK)

    def call(self, prompt: str, system_prompt: Optional[str] = None,
             max_tokens: Optional[int] = None) -> LLMResponse:
        import openai  # optional dependency; only a run needs it

        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()

        # vLLM-specific sampling parameters travel outside the OpenAI schema.
        extra_body = {}
        if self.config.top_k > 0:
            extra_body["top_k"] = self.config.top_k
        if self.config.min_p > 0:
            extra_body["min_p"] = self.config.min_p
        extra_body["chat_template_kwargs"] = {
            "enable_thinking": bool(self.config.enable_thinking)}

        req_max_tokens = max_tokens or self.config.max_tokens

        def _do_call(msgs, mt):
            seed_kwargs = {"seed": self.config.seed} if self.config.seed is not None else {}
            return client.chat.completions.create(
                model=self.config.model_id,
                messages=msgs,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=mt,
                extra_body=extra_body if extra_body else None,
                **seed_kwargs,
            )

        try:
            response = _do_call(messages, req_max_tokens)
        except openai.BadRequestError as e:
            err_msg = str(e)

            # Shape 1: "max_tokens must be at least 1, got -N". The prompt alone
            # is N tokens past the window.
            m = re.search(r"max_tokens must be at least 1, got (-?\d+)", err_msg)
            if m:
                overflow_tokens = abs(int(m.group(1)))
                chars_to_cut = self._chars_to_cut(overflow_tokens)
                messages = self._truncate_user_message(messages, chars_to_cut)
                logger.warning(
                    "Input exceeds context by ~%d tokens, truncating "
                    "~%d chars and retrying with max_tokens=%d",
                    overflow_tokens, chars_to_cut, RETRY_MAX_TOKENS)
                response = _do_call(messages, RETRY_MAX_TOKENS)
            else:
                # Shape 2, in two wordings. The older one gives the total
                # directly; the newer one reports the requested output and the
                # input separately, so the total has to be assembled.
                ctx_len = total = None
                m2 = re.search(
                    r"maximum context length is (\d+).*resulted in (\d+)", err_msg)
                if m2:
                    ctx_len, total = int(m2.group(1)), int(m2.group(2))
                else:
                    m2b = re.search(
                        r"maximum context length is (\d+).*?"
                        r"requested (\d+) output.*?"
                        r"contains at least (\d+) input",
                        err_msg, re.DOTALL)
                    if m2b:
                        ctx_len = int(m2b.group(1))
                        total = int(m2b.group(3)) + int(m2b.group(2))

                if ctx_len is None:
                    raise

                input_tokens = total - req_max_tokens
                clamped = max(128, ctx_len - input_tokens)
                if clamped < req_max_tokens:
                    logger.warning(
                        "Clamping max_tokens %d -> %d (ctx=%d, input=%d)",
                        req_max_tokens, clamped, ctx_len, input_tokens)
                    response = _do_call(messages, clamped)
                else:
                    # The clamp did not bind, so the input alone is over the
                    # window and there is nothing to clamp: truncate instead.
                    overflow = input_tokens - ctx_len
                    chars_to_cut = self._chars_to_cut(overflow)
                    messages = self._truncate_user_message(messages, chars_to_cut)
                    logger.warning(
                        "Input (%d) exceeds context (%d), truncating "
                        "~%d chars and retrying with max_tokens=%d",
                        input_tokens, ctx_len, chars_to_cut, RETRY_MAX_TOKENS)
                    response = _do_call(messages, RETRY_MAX_TOKENS)

        latency_ms = (time.time() - start_time) * 1000

        msg = response.choices[0].message
        full_content = msg.content or ""

        # Three routes to the chain of thought, in the order they take
        # precedence:
        #   1. vLLM 0.19+ with --reasoning-config puts it in a "reasoning" key
        #      of the raw message dict,
        #   2. vLLM with --reasoning-parser exposes it as msg.reasoning_content,
        #   3. no parser at all leaves <think>...</think> inline in the content.
        # In routes 1 and 2 the content is already the answer alone.
        reasoning_content = None
        content = full_content

        raw_msg = response.model_dump()["choices"][0]["message"]
        if raw_msg.get("reasoning"):
            reasoning_content = raw_msg["reasoning"]
        elif getattr(msg, "reasoning_content", None):
            reasoning_content = msg.reasoning_content
        elif "</think>" in full_content:
            parts = full_content.split("</think>", 1)
            reasoning_content = parts[0].replace("<think>", "").strip()
            content = parts[1].strip()

        return LLMResponse(
            content=content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": (response.usage.completion_tokens
                                      if response.usage else 0),
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            latency_ms=latency_ms,
            raw_response={
                "full_text": full_content,
                "reasoning": reasoning_content,
                "final_answer": content,
            },
        )


class OpenAICompatibleClient(BaseClient):
    """Any OpenAI-schema endpoint reached over the network.

    Used for the frontier-API probe in the appendix, where the provider is
    selected by ``base_url`` and the key by ``api_key_env``; DeepSeek is
    ``api_key_env="DEEPSEEK_API_KEY"`` with its own base URL. No number in the
    main tables comes through here.
    """

    def __init__(self, model_config: ModelConfig,
                 base_url: Optional[str] = None,
                 api_key_env: str = "OPENAI_API_KEY"):
        super().__init__(model_config)
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(f"{api_key_env} is not set in the environment")

    def call(self, prompt: str, system_prompt: Optional[str] = None,
             max_tokens: Optional[int] = None) -> LLMResponse:
        import openai

        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()
        response = client.chat.completions.create(
            model=self.config.model_id,
            messages=messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=max_tokens or self.config.max_tokens,
        )
        latency_ms = (time.time() - start_time) * 1000

        full_content = response.choices[0].message.content or ""
        content, reasoning_content = full_content, None
        # Route 3 again: a hosted thinking model that returns its trace inline.
        if "</think>" in full_content:
            parts = full_content.split("</think>", 1)
            reasoning_content = parts[0].replace("<think>", "").strip()
            content = parts[1].strip()

        return LLMResponse(
            content=content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": (response.usage.completion_tokens
                                      if response.usage else 0),
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            latency_ms=latency_ms,
            raw_response={
                "full_text": full_content,
                "reasoning": reasoning_content,
                "final_answer": content,
            },
        )


class GoogleClient(BaseClient):
    """Gemini through google-genai, for the frontier-API probe."""

    MAX_RETRIES = 3

    def __init__(self, model_config: ModelConfig,
                 api_key_env: str = "GOOGLE_API_KEY"):
        super().__init__(model_config)
        self.api_key_env = api_key_env
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(f"{api_key_env} is not set in the environment")

    def call(self, prompt: str, system_prompt: Optional[str] = None,
             max_tokens: Optional[int] = None) -> LLMResponse:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        start_time = time.time()
        # Rate limiting is the common failure and it is transient, so back off
        # 1s, 2s, 4s before giving up.
        for attempt in range(self.MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=self.config.model_id,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        max_output_tokens=max_tokens or self.config.max_tokens,
                    ),
                )
                content = response.text
                if content is None:
                    # A blocked candidate returns no text; surface why instead
                    # of recording an empty verdict.
                    candidates = getattr(response, "candidates", None)
                    if candidates and hasattr(candidates[0], "finish_reason"):
                        raise ValueError(
                            f"Response blocked: {candidates[0].finish_reason}")
                    raise ValueError("Empty response from API")
                break
            except Exception:
                if attempt == self.MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)

        latency_ms = (time.time() - start_time) * 1000
        usage = response.usage_metadata
        return LLMResponse(
            content=content,
            model=self.config.model_id,
            usage={
                "prompt_tokens": usage.prompt_token_count if usage else 0,
                "completion_tokens": usage.candidates_token_count if usage else 0,
                "total_tokens": usage.total_token_count if usage else 0,
            },
            latency_ms=latency_ms,
            raw_response={"full_text": content, "reasoning": None,
                          "final_answer": content},
        )


def client_for(model_config: ModelConfig, base_url: Optional[str] = None,
               **kwargs) -> BaseClient:
    """Build the client one :class:`ModelConfig` asks for."""
    if model_config.api_type == "vllm":
        return VLLMClient(model_config, base_url=base_url, **kwargs)
    if model_config.api_type == "openai":
        return OpenAICompatibleClient(model_config, base_url=base_url, **kwargs)
    if model_config.api_type == "google":
        return GoogleClient(model_config, **kwargs)
    raise ValueError(f"unknown api_type {model_config.api_type!r}")


def get_client(model: str, base_url: Optional[str] = None,
               model_id: Optional[str] = None,
               seed: Optional[int] = None) -> BaseClient:
    """Client for one of the evaluated models, by paper name.

    ``model_id`` overrides the served model name, for a server started with a
    different ``--served-model-name``.
    ``seed`` sets the vLLM request's sampling seed; it does not guarantee
    deterministic results across server versions or batching configurations.
    """
    if model not in MODELS:
        raise ValueError(
            f"unknown model {model!r}; the evaluated models are "
            f"{list(MODELS)}")
    cfg = MODELS[model]
    cfg = replace(cfg, model_id=model_id or cfg.model_id, seed=seed)
    return client_for(cfg, base_url=base_url)
