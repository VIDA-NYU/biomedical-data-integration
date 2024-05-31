import pandas as pd
from typing import List


class ValueMapper:
    """
    A ValueMapper represents objects that transform the values in a input
    column to the values from a new output column.
    """

    def map(self, input_column: pd.Series) -> pd.Series:
        """
        Every concrete ValueMapper should implement this method, which takes a
        pandas Series as input and returns a new pandas Series with transformed
        values.
        """
        pass


class IdentityValueMapper(ValueMapper):
    """
    A column mapper that maps each value in input column into itself.
    """

    def map(self, input_column: pd.Series) -> pd.Series:
        """
        Simply copies the values in input_column to the output column.
        """
        return input_column.copy()


class FunctionValueMapper(ValueMapper):
    """
    A column mapper that transforms each value in the input column using the
    provided custom function.
    """

    def __init__(self, function):
        self.function = function

    def map(self, input_column: pd.Series) -> pd.Series:
        """
        Applies the given function to each value in input_column to generate
        the output column.
        """
        return input_column.map(self.function)


class DictionaryMapper(ValueMapper):
    """
    A column mapper that transforms each value in the input column using the
    values stored in the provided dictionary.
    """

    def __init__(self, dictionary: dict):
        self.dictionary = dictionary

    def map(self, input_column: pd.Series) -> pd.Series:
        """
        Transforms the values in the input_column to the values specified in
        the dictionary provided using the object constructor.
        """
        return input_column.map(self.dictionary)


def map_column_values(
    input_column: pd.Series, target: str, value_mapper: ValueMapper
) -> pd.Series:
    new_column = value_mapper.map(input_column)
    new_column.name = target
    return new_column


def materialize_mapping(
    input_dataframe: pd.DataFrame, target: List[dict]
) -> pd.DataFrame:
    output_dataframe = pd.DataFrame()
    for mapping_spec in target:
        from_column_name = mapping_spec["from"]
        to_column_name = mapping_spec["to"]
        value_mapper = mapping_spec["mapper"]
        output_dataframe[to_column_name] = map_column_values(
            input_dataframe[from_column_name], to_column_name, value_mapper
        )
    return output_dataframe
