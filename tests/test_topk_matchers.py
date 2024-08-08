import pandas as pd
from bdikit.mapping_algorithms.column_mapping.topk_matchers import (
    TopkColumnMatcher,
    CLTopkColumnMatcher,
)


def test_basic_column_mapping_algorithms():

    topk_matcher = CLTopkColumnMatcher()

    # given
    table1 = pd.DataFrame({"col_1": ["a1", "b1", "c1"], "col_2": ["a2", "b2", "c2"]})
    table2 = pd.DataFrame(
        {
            "target_col_1": ["a1", "b1", "c1"],
            "target_col_2": ["a2", "b2", "c2"],
            "target_col_3": ["a1", "b1", "c1"],
            "target_col_4": ["a2", "b2", "c2"],
        }
    )

    # when
    matches = topk_matcher.get_recommendations(source=table1, target=table2, top_k=3)
    # matches = topk_matcher.get_recommendations(source=table1[["col_1"]], target=table2, top_k=3)

    print(f"Matches: {matches}")
    # then
    assert isinstance(matches, list)
    # assert len(matches) == 2

    assert matches[0]["source_column"] == "col_1"
    col1_matches = matches[0]["top_k_columns"]
    assert len(col1_matches) == 3

    # assert matches[1]["source_column"] == "col_2"
    # col2_matches = matches[1]["top_k_columns"]
    # assert len(col2_matches) == 3


    


# [
#     {
#         "source_column": "col_1",
#         "top_k_columns": [
#             ColumnScore(column_name="target_col_1", score=0.8359868),
#             ColumnScore(column_name="target_col_3", score=0.7979812),
#             ColumnScore(column_name="target_col_2", score=0.7916803),
#         ],
#     },
#     {
#         "source_column": "col_2",
#         "top_k_columns": [
#             ColumnScore(column_name="target_col_2", score=0.83525527),
#             ColumnScore(column_name="target_col_4", score=0.8028608),
#             ColumnScore(column_name="target_col_1", score=0.80242705),
#         ],
#     },
# ]
