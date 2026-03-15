from collections.abc import Sequence
from typing import Protocol, cast

from dify_graph.file.models import File
from dify_graph.model_runtime.entities import PromptMessageRole
from dify_graph.model_runtime.entities.message_entities import (
    ImagePromptMessageContent,
    PromptMessage,
    TextPromptMessageContent,
)
from dify_graph.model_runtime.entities.model_entities import AIModelEntity
from dify_graph.model_runtime.memory import PromptMessageMemory
from dify_graph.model_runtime.model_providers.__base.large_language_model import LargeLanguageModel
from dify_graph.runtime import VariablePool
from dify_graph.variables.segments import ArrayAnySegment, ArrayFileSegment, FileSegment, NoneSegment

from .exc import InvalidVariableTypeError


class _GraphPreparedLLM(Protocol):
    def get_model_schema(self) -> AIModelEntity: ...


class _LegacyModelInstance(Protocol):
    model_type_instance: object
    model_name: str
    credentials: object


def fetch_model_schema(*, model_instance: object) -> AIModelEntity:
    get_model_schema = getattr(model_instance, "get_model_schema", None)
    if callable(get_model_schema):
        model_schema = cast(_GraphPreparedLLM, model_instance).get_model_schema()
    else:
        legacy_model_instance = cast(_LegacyModelInstance, model_instance)
        model_schema = cast(LargeLanguageModel, legacy_model_instance.model_type_instance).get_model_schema(
            legacy_model_instance.model_name,
            legacy_model_instance.credentials,
        )
    if not model_schema:
        raise ValueError(f"Model schema not found for {getattr(model_instance, 'model_name', 'unknown model')}")
    return model_schema


def fetch_files(variable_pool: VariablePool, selector: Sequence[str]) -> Sequence["File"]:
    variable = variable_pool.get(selector)
    if variable is None:
        return []
    elif isinstance(variable, FileSegment):
        return [variable.value]
    elif isinstance(variable, ArrayFileSegment):
        return variable.value
    elif isinstance(variable, NoneSegment | ArrayAnySegment):
        return []
    raise InvalidVariableTypeError(f"Invalid variable type: {type(variable)}")


def convert_history_messages_to_text(
    *,
    history_messages: Sequence[PromptMessage],
    human_prefix: str,
    ai_prefix: str,
) -> str:
    string_messages: list[str] = []
    for message in history_messages:
        if message.role == PromptMessageRole.USER:
            role = human_prefix
        elif message.role == PromptMessageRole.ASSISTANT:
            role = ai_prefix
        else:
            continue

        if isinstance(message.content, list):
            content_parts = []
            for content in message.content:
                if isinstance(content, TextPromptMessageContent):
                    content_parts.append(content.data)
                elif isinstance(content, ImagePromptMessageContent):
                    content_parts.append("[image]")

            inner_msg = "\n".join(content_parts)
            string_messages.append(f"{role}: {inner_msg}")
        else:
            string_messages.append(f"{role}: {message.content}")

    return "\n".join(string_messages)


def fetch_memory_text(
    *,
    memory: PromptMessageMemory,
    max_token_limit: int,
    message_limit: int | None = None,
    human_prefix: str = "Human",
    ai_prefix: str = "Assistant",
) -> str:
    history_messages = memory.get_history_prompt_messages(
        max_token_limit=max_token_limit,
        message_limit=message_limit,
    )
    return convert_history_messages_to_text(
        history_messages=history_messages,
        human_prefix=human_prefix,
        ai_prefix=ai_prefix,
    )
