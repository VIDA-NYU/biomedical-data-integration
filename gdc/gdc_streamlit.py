import streamlit as st
import pandas as pd
from gdc_api_v2 import GDCSchema
from gdc_candidate_matcher import GDCCandidateMatcher
from gdc_scoring_interface import GPTHelper

st.title('Column Matching with GDC Demo')

st.header("1. Upload the Raw Data", anchor=False)
uploaded_file = st.file_uploader("Upload Your CSV File", type=["csv"])

raw_dataset = None
if uploaded_file:
    st.write("filename:", uploaded_file.name)
    if uploaded_file.type == "text/csv":
        raw_dataset = pd.read_csv(uploaded_file)
        st.dataframe(raw_dataset)

st.header("2. Explore GDC Schema", anchor=False)
schema = GDCSchema()
schema_dfs = schema.parse_schema_to_df()


subschema = st.selectbox(
    'Select a subschema: ',
    list(schema_dfs.keys()))

def dataframe_with_selections(df: pd.DataFrame, init_value: bool = False) -> pd.DataFrame:
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", init_value)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)


selection = dataframe_with_selections(schema_dfs[subschema])

if "selected_labels" not in st.session_state:
    st.session_state.selected_labels = set()
if "gdc_labels" not in st.session_state:
    st.session_state.gdc_labels = set()

st.session_state.selected_labels.update(selection["column_name"].to_list())
st.write(st.session_state.selected_labels)
if st.button('Add to GDC labels'):
    st.session_state.gdc_labels.update(st.session_state.selected_labels)
if st.button('Clear selection'):
    st.session_state.selected_labels = set()

st.write("Selected labels:")
st.write(st.session_state.gdc_labels)

st.header("3. Match Columns Using CTA", anchor=False)
gpt = GPTHelper(api_key="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")


output_dict = {"column_name": [], "generated_column_type": []}
if st.button('Ask CTA'):
    if raw_dataset is None:
        st.warning("Please upload a CSV file to proceed.")
    else:
        for col_name in raw_dataset.columns:
            result = gpt.ask_cta(labels=list(st.session_state.gdc_labels), context=col_name)
            output_dict["column_name"].append(col_name)
            output_dict["generated_column_type"].append(result)

st.dataframe(pd.DataFrame(output_dict))
        