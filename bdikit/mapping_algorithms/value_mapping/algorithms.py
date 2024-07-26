from typing import List, NamedTuple, Callable, Tuple, Union
import ast
from openai import OpenAI
from polyfuzz import PolyFuzz
from polyfuzz.models import EditDistance, TFIDF, Embeddings
from flair.embeddings import TransformerWordEmbeddings, WordEmbeddings
from rapidfuzz import fuzz
from autofj import AutoFJ
from Levenshtein import ratio
import pandas as pd
import flair
import torch
from bdikit.config import get_device, VALUE_MATCHING_THRESHOLD

flair.device = torch.device(get_device())


class ValueMatch(NamedTuple):
    """
    Represents a match between a source value and a target value with a
    similarity score.
    """

    source_value: str
    target_value: str
    similarity: float


class BaseValueMatcher:
    """
    Base class for value matching algorithms, i.e., algorithms that match
    values from a source domain to values from a target domain.
    """

    def match(
        self, source_values: List[str], target_values: List[str]
    ) -> List[ValueMatch]:
        raise NotImplementedError("Subclasses must implement this method")


class PolyFuzzValueMatcher(BaseValueMatcher):
    """
    Base class for value matching algorithms based on the PolyFuzz library.
    """

    def __init__(self, polyfuzz_model: PolyFuzz, threshold: float):
        self.model = polyfuzz_model
        self.threshold = threshold

    def match(
        self,
        source_values: List[str],
        target_values: List[str],
    ) -> List[ValueMatch]:

        self.model.match(source_values, target_values)
        match_results = self.model.get_matches()
        match_results.sort_values(by="Similarity", ascending=False, inplace=True)

        matches = []
        for _, row in match_results.iterrows():
            source_value = row["From"]
            target_value = row["To"]
            similarity = row["Similarity"]
            if similarity >= self.threshold:
                matches.append(ValueMatch(source_value, target_value, similarity))

        return matches


class TFIDFValueMatcher(PolyFuzzValueMatcher):
    """
    Value matching algorithm based on the TF-IDF similarity between values.
    """

    def __init__(
        self,
        n_gram_range: Tuple[int, int] = (1, 3),
        clean_string: bool = True,
        threshold: float = VALUE_MATCHING_THRESHOLD,
        top_n: int = 1,
        cosine_method: str = "sparse",
    ):

        super().__init__(
            PolyFuzz(
                method=TFIDF(
                    n_gram_range=n_gram_range,
                    clean_string=clean_string,
                    min_similarity=threshold,
                    top_n=top_n,
                    cosine_method=cosine_method,
                )
            ),
            threshold,
        )


class EditDistanceValueMatcher(PolyFuzzValueMatcher):
    """
    Value matching algorithm based on the edit distance between values.
    """

    def __init__(
        self,
        scorer: Callable[[str, str], float] = fuzz.ratio,
        n_jobs: int = -1,
        threshold: float = VALUE_MATCHING_THRESHOLD,
    ):
        # Return scores between 0 and 1
        normalized_scorer = lambda str1, str2: scorer(str1, str2) / 100.0
        super().__init__(
            PolyFuzz(
                method=EditDistance(
                    n_jobs=n_jobs, scorer=normalized_scorer, normalize=False
                )
            ),
            threshold,
        )


class EmbeddingValueMatcher(PolyFuzzValueMatcher):
    """
    Value matching algorithm based on the cosine similarity of value embeddings.
    """

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        threshold: float = VALUE_MATCHING_THRESHOLD,
        top_n: int = 1,
        cosine_method: str = "sparse",
    ):
        embeddings = TransformerWordEmbeddings(model_name)
        method = Embeddings(
            embeddings,
            min_similarity=threshold,
            top_n=top_n,
            cosine_method=cosine_method,
        )
        super().__init__(PolyFuzz(method), threshold)


class FastTextValueMatcher(PolyFuzzValueMatcher):
    """
    Value matching algorithm based on the cosine similarity of FastText embeddings.
    """

    def __init__(
        self,
        model_name: str = "en-crawl",
        threshold: float = VALUE_MATCHING_THRESHOLD,
        top_n: int = 1,
        cosine_method: str = "sparse",
    ):
        embeddings = WordEmbeddings(model_name)
        method = Embeddings(
            embeddings,
            min_similarity=threshold,
            top_n=top_n,
            cosine_method=cosine_method,
        )
        super().__init__(PolyFuzz(method), threshold)


class GPTValueMatcher(BaseValueMatcher):
    def __init__(
        self,
        threshold: float = VALUE_MATCHING_THRESHOLD,
    ):
        self.client = OpenAI()
        self.threshold = threshold

    def match(
        self,
        source_values: List[str],
        target_values: List[str],
    ) -> List[ValueMatch]:
        target_values_set = set(target_values)
        matches = []

        for source_value in source_values:
            completion = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an intelligent system that given a term, you have to choose a value from a list that best matches the term. "
                        "These terms belong to the medical domain, and the list contains terms in the Genomics Data Commons (GDC) format.",
                    },
                    {
                        "role": "user",
                        "content": f'For the term: "{source_value}", choose a value from this list {target_values}. '
                        "Return the value from the list with a similarity score, between 0 and 1, with 1 indicating the highest similarity. "
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. "
                        'Only provide a Python dictionary. For example {"term": "term from the list", "score": 0.8}.',
                    },
                ],
            )

            response_message = completion.choices[0].message.content
            try:
                response_dict = ast.literal_eval(response_message)
                target_value = response_dict["term"]
                score = float(response_dict["score"])
                if target_value in target_values_set and score >= self.threshold:
                    matches.append(ValueMatch(source_value, target_value, score))
            except:
                print(
                    f'Errors parsing response for "{source_value}": {response_message}'
                )

        return matches


class AutoFuzzyJoinValueMatcher(BaseValueMatcher):
    def __init__(
        self,
        threshold: float = VALUE_MATCHING_THRESHOLD,
    ):
        self.threshold = threshold

    def match(
        self,
        source_values: List[str],
        target_values: List[str],
    ) -> List[ValueMatch]:

        source_values = sorted(list(set(source_values)))
        target_values = sorted(list(set(target_values)))

        df_source_values = pd.DataFrame(
            {"id": range(1, len(source_values) + 1), "title": source_values}
        )
        df_target_values = pd.DataFrame(
            {"id": range(1, len(target_values) + 1), "title": target_values}
        )

        matches = []
        try:
            autofj = AutoFJ(
                precision_target=self.threshold,
                join_function_space="autofj_md",
                verbose=True,
            )
            LR_joins = autofj.join(df_source_values, df_target_values, id_column="id")
            if len(LR_joins) > 0:
                for _, row in LR_joins.iterrows():
                    title_l = row["title_l"]
                    title_r = row["title_r"]
                    similarity = ratio(title_l, title_r)
                    if similarity >= self.threshold:
                        matches.append(ValueMatch(title_l, title_r, similarity))
        except Exception as e:
            return matches
        return matches
