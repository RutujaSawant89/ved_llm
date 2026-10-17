"""TrustLM RAG — Prompt Builder.

Constructs the final prompt string that is sent to the generator.

Why a dedicated PromptBuilder?
  - The prompt template is the most critical point where retrieval meets
    generation.  A separate, testable module makes it easy to:
    * Iterate on the prompt format without touching pipeline logic.
    * Swap templates for different use cases (QA, summarisation, etc.).
    * Later add system instructions for Verification/Trust modules.

Prompt structure:
    [System instruction]
    You are TrustLM, a reliable AI assistant.
    Answer using ONLY the provided context.
    If the context doesn't have enough information, say so explicitly.
    Do not invent facts.

    [Context]
    <formatted context from ContextBuilder>

    [Question]
    <user query>

    [Answer]
    <generator fills this in>
"""

from __future__ import annotations

from typing import List, Optional


# System instruction prefix that steers the generator toward grounded responses.
# "Only use provided context" is fundamental for a RAG pipeline where we want
# answers to be traceable to retrieved documents.
_SYSTEM_INSTRUCTION = """\
You are TrustLM, a trustworthy academic assistant.

Answer the user's question using ONLY the provided retrieved evidence.

Rules:
1. Do not invent facts.
2. Do not use unsupported external knowledge.
3. If evidence is insufficient, explicitly say:
   "The provided notes do not contain sufficient information to answer this reliably."
4. Do not pretend unsupported information is present in the notes.
5. For comparisons, organize information clearly in a table when appropriate.
6. For multi-step/process questions, use numbered steps.
7. Preserve terminology used in the source notes.
8. Every important factual claim must be traceable to retrieved evidence.
9. Mention the source/module/slide where appropriate.
"""


class PromptBuilder:
    """Build the prompt string sent to the generator.

    Args:
        system_instruction: Override the default system instruction.
        answer_prefix:      String placed at the end to prime the generator
                            (e.g. "Answer:").
    """

    def __init__(
        self,
        system_instruction: Optional[str] = None,
        answer_prefix: str = "Answer:",
    ) -> None:
        self.system_instruction = system_instruction or _SYSTEM_INSTRUCTION
        self.answer_prefix = answer_prefix

    def build_prompt(
        self,
        query: str,
        context: str,
    ) -> str:
        """Construct the complete prompt for the generator.

        Args:
            query:   The user's question.
            context: The formatted context string from ContextBuilder.

        Returns:
            Full prompt string including system instruction, context, and question.

        Raises:
            ValueError: If query or context is empty.
        """
        if not query or not query.strip():
            raise ValueError("[PROMPT] Query cannot be empty.")

        if not context or not context.strip():
            # It's valid to have no context (empty knowledge base); warn instead of crash.
            context = "[No relevant context was retrieved from the knowledge base.]"

        prompt = (
            f"{self.system_instruction.strip()}\n\n"
            f"Context:\n{context.strip()}\n\n"
            f"Question:\n{query.strip()}\n\n"
            f"{self.answer_prefix}"
        )
        return prompt

    def build_no_context_prompt(self, query: str) -> str:
        """Build a prompt when no relevant context was retrieved.

        This variant makes it explicit to the generator that no supporting
        documents were found, so it should acknowledge that clearly.

        Args:
            query: The user's question.

        Returns:
            Prompt string indicating empty knowledge base.
        """
        return self.build_prompt(
            query=query,
            context="[No relevant context was retrieved from the knowledge base.]",
        )
