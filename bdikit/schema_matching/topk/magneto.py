import pandas as pd
from typing import Dict, Any, List
from magneto import Magneto as Magneto_Lib
from bdikit.schema_matching.one2one.base import BaseSchemaMatcher
from bdikit.download import get_cached_model_or_download
from bdikit.schema_matching.topk.base import (
    ColumnScore,
    TopkMatching,
    BaseTopkSchemaMatcher,
)

DEFAULT_MAGNETO_MODEL = "magneto-gdc-v0.1"


class MagnetoBase(BaseSchemaMatcher, BaseTopkSchemaMatcher):
    def __init__(self, kwargs: Dict[str, Any] = None):
        if kwargs is None:
            kwargs = {}
        self.magneto = Magneto_Lib(**kwargs)

    def map(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
    ):
        # There is an issue in Magneto to get the top-1 match, so get top 2 and then filter
        self.magneto.params["topk"] = 2  # Magneto does not provide a method to set topk
        raw_matches = self.magneto.get_matches(source, target)

        # Organizing data into the desired structure
        sorted_dict = {}
        for (source, target), score in raw_matches.items():
            source_column = source[1]
            target_column = target[1]
            if source_column not in sorted_dict:
                sorted_dict[source_column] = []
            sorted_dict[source_column].append((target_column, score))

        # Sorting the lists by value in descending order and get top 1
        formatted_matches = {}
        for key in sorted_dict:
            sorted_matches = sorted(sorted_dict[key], key=lambda x: x[1], reverse=True)
            formatted_matches[key] = sorted_matches[0][0]

        return formatted_matches

    def get_recommendations(
        self, source: pd.DataFrame, target: pd.DataFrame, top_k: int
    ) -> List[TopkMatching]:
        self.magneto.params["topk"] = (
            top_k  # Magneto does not provide a method to set topk
        )
        raw_matches = self.magneto.get_matches(source, target)

        # Organizing data into the desired structure
        sorted_dict = {}
        for (source, target), score in raw_matches.items():
            source_column = source[1]
            target_column = target[1]
            if source_column not in sorted_dict:
                sorted_dict[source_column] = []
            sorted_dict[source_column].append((target_column, score))

        # Sorting the lists by value in descending order and format top k
        top_k_results = []
        for key in sorted_dict:
            sorted_matches = sorted(sorted_dict[key], key=lambda x: x[1], reverse=True)
            top_k_columns = [ColumnScore(name, score) for name, score in sorted_matches]
            top_k_results.append(
                {
                    "source_column": [key] * len(top_k_columns),
                    "top_k_columns": top_k_columns,
                }
            )

        return top_k_results


class Magneto(MagnetoBase):
    def __init__(self):
        super().__init__()


class MagnetoFT(MagnetoBase):
    def __init__(
        self,
        encoding_mode: str = "header_values_verbose",
        model_name: str = DEFAULT_MAGNETO_MODEL,
        model_path: str = None,
    ):
        embedding_model = check_magneto_model(model_name, model_path)
        kwargs = {"encoding_mode": encoding_mode, "embedding_model": embedding_model}
        super().__init__(kwargs)


class MagnetoGPT(MagnetoBase):
    def __init__(self):
        kwargs = {"use_bp_reranker": False, "use_gpt_reranker": True}
        super().__init__(kwargs)


class MagnetoFTGPT(MagnetoBase):
    def __init__(
        self,
        encoding_mode: str = "header_values_verbose",
        model_name: str = DEFAULT_MAGNETO_MODEL,
        model_path: str = None,
    ):
        embedding_model = check_magneto_model(model_name, model_path)
        kwargs = {
            "encoding_mode": encoding_mode,
            "embedding_model": embedding_model,
            "use_bp_reranker": False,
            "use_gpt_reranker": True,
        }
        super().__init__(kwargs)


def check_magneto_model(model_name: str, model_path: str):
    if model_name and model_path:
        raise ValueError(
            "Only one of model_name or model_path should be provided "
            "(they are mutually exclusive)"
        )

    if model_path:
        return model_path
    elif model_name:
        return get_cached_model_or_download(model_name)
    else:
        raise ValueError("Either model_name or model_path must be provided")
