import os
from typing import List, Dict, Tuple, Optional
from bdikit.config import get_device
import numpy as np
import pandas as pd
import torch
from bdikit.models.contrastive_learning.cl_models import (
    BarlowTwinsSimCLR,
)
from bdikit.models.contrastive_learning.cl_pretrained_dataset import (
    PretrainTableDataset,
)
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from bdikit.download import get_cached_model_or_download
from bdikit.utils import check_gdc_cache, write_embeddings_to_cache


DEFAULT_CL_MODEL = "bdi-cl-v0.2"


class ContrastiveLearningAPI:
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        batch_size: int = 128,
    ):
        if model_name and model_path:
            raise ValueError(
                "Only one of model_name or model_path should be provided "
                "(they are mutually exclusive)"
            )

        if model_path:
            self.model_path = model_path
        elif model_name:
            self.model_path = get_cached_model_or_download(model_name)
        else:
            raise ValueError("Either model_name or model_path must be provided")

        self.unlabeled = PretrainTableDataset()
        self.batch_size = batch_size
        self.device = get_device()
        self.model = self.load_checkpoint()

    def load_checkpoint(self, lm: str = "roberta"):
        ckpt = torch.load(self.model_path, map_location=torch.device("cpu"))
        scale_loss = 0.1
        lambd = 3.9
        model = BarlowTwinsSimCLR(scale_loss, lambd, device=self.device, lm=lm)
        model = model.to(self.device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model

    def get_embeddings(self, table: pd.DataFrame) -> List[np.ndarray]:
        return self._load_table_tokens(table)

    def get_recommendations(
        self, table: pd.DataFrame, target: pd.DataFrame, top_k: int = 10
    ) -> Tuple[List, List[Dict]]:

        l_features = self._load_table_tokens(table)
        r_features = self._load_table_tokens(target)
        cosine_sim = cosine_similarity(l_features, r_features)

        # print(f"l_features - {len(l_features)}:{l_features[0].shape}\nr-feature - {len(r_features)}:{r_features[0].shape}\nCosine - {cosine_sim.shape}")

        top_k_results = []
        l_column_ids = table.columns
        gt_column_ids = target.columns

        for index, similarities in enumerate(cosine_sim):
            top_k_indices = np.argsort(similarities)[::-1][:top_k]
            top_k_column_names = [gt_column_ids[i] for i in top_k_indices]
            top_k_similarities = [str(round(similarities[i], 4)) for i in top_k_indices]
            top_k_columns = list(zip(top_k_column_names, top_k_similarities))
            result = {
                "Candidate column": l_column_ids[index],
                "Top k columns": top_k_columns,
            }
            top_k_results.append(result)
        recommendations = self._extract_recommendations_from_top_k(top_k_results)
        return recommendations, top_k_results

    def _extract_recommendations_from_top_k(self, top_k_results: List[dict]):
        recommendations = set()
        for result in top_k_results:
            for name, _ in result["Top k columns"]:
                recommendations.add(name)
        return list(recommendations)

    def _sample_to_15_rows(self, table: pd.DataFrame):
        if len(table) > 15:
            unique_rows = table.drop_duplicates()
            num_unique_rows = len(unique_rows)
            if num_unique_rows <= 15:
                needed_rows = 15 - num_unique_rows
                additional_rows = table[~table.index.isin(unique_rows.index)].sample(
                    n=needed_rows, replace=True, random_state=1
                )
                table = pd.concat([unique_rows, additional_rows])
            else:
                table = unique_rows.sample(n=15, random_state=1)
        return table

    def _load_table_tokens(self, table: pd.DataFrame) -> List[np.ndarray]:

        embedding_file, embeddings = check_gdc_cache(table, self.model_path)

        if embeddings != None:
            print(f"Table features loaded for {len(table.columns)} columns")
            return embeddings

        tables = []
        for _, column in enumerate(table.columns):
            curr_table = pd.DataFrame(table[column])
            curr_table = self._sample_to_15_rows(curr_table)
            tables.append(curr_table)
        vectors = self._inference_on_tables(tables)
        print(f"Table features extracted from {len(table.columns)} columns")
        embeddings = [vec[-1] for vec in vectors]

        if embedding_file != None:
            write_embeddings_to_cache(embedding_file, embeddings)

        return embeddings

    def _inference_on_tables(self, tables: List[pd.DataFrame]) -> List[List]:
        total = len(tables)
        batch = []
        results = []

        for tid, table in tqdm(enumerate(tables), total=total):
            x, _ = self.unlabeled._tokenize(table)
            batch.append((x, x, []))

            if tid == total - 1 or len(batch) == self.batch_size:
                with torch.no_grad():
                    x, _, _ = self.unlabeled.pad(batch)
                    column_vectors = self.model.inference(x)
                    ptr = 0
                    for xi in x:
                        current = []
                        for token_id in xi:
                            if token_id == self.unlabeled.tokenizer.cls_token_id:
                                current.append(column_vectors[ptr].cpu().numpy())
                                ptr += 1
                        results.append(current)
                batch.clear()
        return results
