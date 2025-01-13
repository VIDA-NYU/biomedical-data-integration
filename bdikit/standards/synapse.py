import json
import pandas as pd
from os.path import join, dirname
from typing import List, Dict
from bdikit.standards.base import BaseStandard


SYNAPSE_SCHEMA_PATH = join(dirname(__file__), "../resource/synapse_schema.json")


class Synapse(BaseStandard):
    """
    Class for Synapse standard.
    """

    def __init__(self) -> None:
        self.data = None
        self.__read_data()

    def __read_data(self):
        with open(SYNAPSE_SCHEMA_PATH) as json_file:
            self.data = json.load(json_file)

    def get_columns(self) -> List[str]:
        return list(self.data.keys())

    def get_column_values(
        self, column_names: List[str]
    ) -> Dict[str, List]:  # get_gdc_data
        column_values = {}

        for column_name in column_names:
            raw_metadata = self.data.get(column_name, {})
            column_values[column_name] = list(raw_metadata.get("value_data", {}).keys())

        return column_values

    def get_column_metadata(
        self, column_names: List[str]
    ) -> Dict[str, Dict]:  # get_gdc_metadata
        column_metadata = {}

        for column_name in column_names:
            raw_metadata = self.data.get(column_name, {})
            column_metadata[column_name] = {}
            column_metadata[column_name]["description"] = raw_metadata.get(
                "column_description", ""
            )
            column_metadata[column_name]["value_names"] = list(
                raw_metadata.get("value_data", {}).keys()
            )
            column_metadata[column_name]["value_descriptions"] = list(
                raw_metadata.get("value_data", {}).values()
            )

        return column_metadata

    def get_dataframe_rep(self) -> pd.DataFrame:
        reshaped_data = {
            key: list(value["value_data"].keys()) for key, value in self.data.items()
        }

        # Ensure all lists have the same length by padding with None
        max_length = max(len(v) for v in reshaped_data.values())
        for k, v in reshaped_data.items():
            reshaped_data[k].extend([None] * (max_length - len(v)))

        df = pd.DataFrame.from_dict(reshaped_data, orient="columns")

        return df
