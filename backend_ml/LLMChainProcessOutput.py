from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from langchain.chains import LLMChain

from langchain.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainRun,
    CallbackManager,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain.chains.base import Chain
from langchain.load.dump import dumpd
from langchain.prompts.prompt import PromptTemplate
from langchain.pydantic_v1 import Extra, Field
from langchain.schema import (
    BaseLLMOutputParser,
    BasePromptTemplate,
    LLMResult,
    PromptValue,
    StrOutputParser,
)
from langchain.schema.language_model import BaseLanguageModel
from langchain.utils.input import get_colored_text

from langchain.schema.output import Generation

class LLMChainProcessOutput(LLMChain):
    def create_outputs(self, llm_result: LLMResult) -> List[Dict[str, Any]]:
        """Create outputs from response."""

        llm_result_processed = LLMResult(llm_output=llm_result.llm_output, run=llm_result.run, generations=[])
        for generation in llm_result.generations:
            llm_result_processed.generations.append([Generation(text=g.text.replace("<|endoftext|></s>", "").replace("-- <|endoftext|>", "").replace("<|endoftext|>","")) 
                                                     for g in generation])

        result = [
            # Get the text of the top generated string.
            {
                self.output_key: self.output_parser.parse_result(generation),
                "full_generation": generation,
            }
            for generation in llm_result_processed.generations
        ]
        if self.return_final_only:
            result = [{self.output_key: r[self.output_key]} for r in result]
        return result
