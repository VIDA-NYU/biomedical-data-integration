import importlib
from enum import Enum
from typing import Mapping, Any
from bdikit.value_matching.base import BaseValueMatcher


class ValueMatchers(Enum):
    TFIDF = ("tfidf", "bdikit.value_matching.polyfuzz.TFIDFValueMatcher")
    EDIT = (
        "edit_distance",
        "bdikit.value_matching.polyfuzz.EditDistanceValueMatcher",
    )
    EMBEDDINGS = (
        "embedding",
        "bdikit.value_matching.polyfuzz.EmbeddingValueMatcher",
    )
    FASTTEXT = (
        "fasttext",
        "bdikit.value_matching.polyfuzz.FastTextValueMatcher",
    )
    GPT = ("gpt", "bdikit.value_matching.gpt.GPTValueMatcher")
    GENE = ("gene", "bdikit.value_matching.gene.Gene")

    def __init__(self, matcher_name: str, matcher_path: str):
        self.matcher_name = matcher_name
        self.matcher_path = matcher_path

    @staticmethod
    def get_matcher(
        matcher_name: str, **matcher_kwargs: Mapping[str, Any]
    ) -> BaseValueMatcher:
        if matcher_name not in matchers:
            names = ", ".join(list(matchers.keys()))
            raise ValueError(
                f"The {matcher_name} algorithm is not supported. "
                f"Supported algorithms are: {names}"
            )
        # Load the class dynamically
        module_path, class_name = matchers[matcher_name].rsplit(".", 1)
        module = importlib.import_module(module_path)

        return getattr(module, class_name)(**matcher_kwargs)


matchers = {method.matcher_name: method.matcher_path for method in ValueMatchers}
