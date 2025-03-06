import argparse
import json
import logging
import tomllib
from datetime import datetime

import streamlit as st
from elasticsearch import Elasticsearch
from openai import OpenAI
from pypdf import PdfReader


logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)
logging.getLogger("elastic_transport").setLevel(logging.WARNING)

today = datetime.today().strftime("%A, %B %d, %Y")

st.set_page_config(
    page_title="GenAI-Powered Intelligence Analysis",
    page_icon="./.streamlit/ce-icon.png",
)

with open(".streamlit/style.html", "r") as f:
    style = f.read()

st.markdown(style, unsafe_allow_html=True)


@st.cache_resource
def connect_es(config: dict) -> Elasticsearch:
    if "ELASTIC_API_KEY" in config:
        try:
            client = Elasticsearch(
                cloud_id=config["ELASTIC_CLOUD_ID"], api_key=config["ELASTIC_API_KEY"]
            )
            # Test the connection
            client.info()
            return client
        except Exception:
            pass
    if "ELASTIC_USER" in config and "ELASTIC_PASSWORD" in config:
        try:
            client = Elasticsearch(
                cloud_id=config["ELASTIC_CLOUD_ID"],
                basic_auth=(config["ELASTIC_USER"], config["ELASTIC_PASSWORD"]),
            )
            # Test the connection
            client.info()
            return client
        except Exception:
            pass
    return False


@st.cache_resource
def connect_llm() -> OpenAI:
    llm_client = OpenAI(
        base_url=config["LLM_PROXY_BASE_URL"],
        api_key=config["LLM_PROXY_API_KEY"],
    )
    try:
        llm_client.models.list()
    except Exception:
        return False
    return llm_client


def read_config(config_path: str) -> dict:
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    return config


def read_file(file_name):
    with open(file_name, "r") as file:
        data = json.load(file)
    return data


def parse_filters(filters: dict) -> dict:
    must = []
    es_filters = []

    # parse date range
    if filters["date_range"] == "Last 30 Days":
        date_filter = {"range": {"date": {"gte": "now-30d"}}}
        must.append(date_filter)
    elif filters["date_range"] == "This Year":
        date_filter = {"range": {"date": {"gte": "2025-01-01"}}}
        must.append(date_filter)

    # parse countries selection
    if len(filters["countries"]) > 0:
        country_filter = {"terms": {"country.name": filters["countries"]}}
        must.append(country_filter)

    # parse classifications selection
    if len(filters["classifications"]) > 0:
        classification_filter = {
            "terms": {"classification": filters["classifications"]}
        }
        must.append(classification_filter)

    # parse sources selection
    if len(filters["sources"]) > 0:
        source_filter = {"terms": {"source": filters["sources"]}}
        must.append(source_filter)

    # parse compartments selection
    if len(filters["compartments"]) > 0:
        compartment_filter = {"terms": {"compartments": filters["compartments"]}}
        must.append(compartment_filter)

    # parse geo
    if filters["geo"]:
        es_filters.append(filters["geo"])

    final_filters = {"bool": {"must": must, "filter": es_filters}}
    return final_filters


def elasticsearch_elser(query_text: str, filters: dict) -> dict:
    logging.info(
        f"Performing Elasticsearch sparse vector query for user search: {query_text}"
    )

    search_filters = parse_filters(filters)
    es_query = {
        "size": 3,
        "retriever": {
            "standard": {
                "query": {
                    "sparse_vector": {
                        "field": "details_embeddings",
                        "inference_id": ".elser_model_2_linux-x86_64",
                        "query": query_text,
                    }
                },
                "filter": search_filters,
            }
        },
    }
    res = es.search(index="intel-reports", body=es_query)
    hits = [hit["_source"] for hit in res["hits"]["hits"]]
    return {"source_docs": hits}


def get_classification_level(classification: str) -> int:
    level_map = {
        "UNCLASSIFIED": 0,
        "HUSH HUSH": 1,
        "SUPER SECRET": 2,
        "ULTRA SUPER SECRET": 3,
    }
    return level_map[classification]


def read_pdf(file, file_upload_progress) -> str:
    full_text = ""
    pdf_reader = PdfReader(file)
    total_pages = len(pdf_reader.pages)

    for i, page in enumerate(pdf_reader.pages):
        full_text += page.extract_text()
        # Update progress bar - convert to percentage (0 to 1)
        progress = (i + 1) / total_pages
        file_upload_progress.progress(progress)

    return full_text


def get_geo_filter(prompt: str, llm_client: OpenAI, llm_model_selection: str) -> dict:
    system_prompt = """
    Task:
    Extract only the geolocation information from the user's query and convert it into an Elasticsearch geo_distance filter. Ignore any non-geolocation content. Focus on identifying the location and distance for the filter. 
    If there isn't any geolocation information in the query, return the word NONE.

    Example:
    User Query: "I want to find all the grocery stores within 2 miles of the Empire State Building."
    Output:
    {"geo_distance": {"distance": "2mi","country.coordinates": {"lat": 40.748817,"lon": -73.985428}}}

    Instructions:
    - Extract the distance and location from the query.
    - Use the location's latitude and longitude.
    - Format the output as a Python dictionary. Don't include anything other than the dictionary object. Don't use triple quotes or the name of the programming language.
    - Be precise and concise.
    """

    response = llm_client.chat.completions.create(
        model=llm_model_selection,
        messages=[{"role": "user", "content": prompt}]
        + [
            {
                "role": "system",
                "content": system_prompt,
            }
        ],
        temperature=0,
    )
    if response.choices[0].finish_reason == "content_filter":
        return None
    elif response.choices[0].message.content == "NONE":
        return None
    else:
        msg = response.choices[0].message.content

    return json.loads(msg)


def main():
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"},
        ]

    # top header
    st.header("GenAI-Powered Intelligence Analysis", divider="grey", anchor=False)
    st.caption("a chatbot powered by Elastic")

    # sidebar
    st.logo(".streamlit/logo-elastic-horizontal-color.png")
    llm_model_selection = st.sidebar.pills(
        label="LLM model selection",
        # label_visibility="hidden",
        options=[
            "gpt-4o",
            "gpt-3.5-turbo",
            "anthropic",
            "cohere",
            "gemini-1.0",
        ],
        default="gpt-4o",
    )
    llm_client = connect_llm()
    st.sidebar.divider()
    rag = st.sidebar.toggle("**RAG with Elastic Vector Search**")
    if rag:
        # st.sidebar.markdown("## Search Filter Options")
        date_range_selection = st.sidebar.pills(
            label="Date Range",
            options=("All Time", "Last 30 Days", "This Year"),
            default="All Time",
        )
        classification_selection = st.sidebar.multiselect(
            label="Classification",
            help="Defaults to All",
            placeholder="Select one or more",
            options=classifications,
        )
        country_selection = st.sidebar.multiselect(
            label="Countries of Interest",
            help="Defaults to All",
            placeholder="Select one or more",
            options=[country["name"] for country in countries],
        )
        source_selection = st.sidebar.multiselect(
            label="Intelligence sources",
            help="Defaults to All",
            placeholder="Select one or more",
            options=sources,
        )
        compartment_selection = st.sidebar.multiselect(
            label="Compartments",
            help="Defaults to All",
            placeholder="Select one or more",
            options=compartments,
        )
        # data feed selection does nothing right now
        data_feed_selection = st.sidebar.pills(
            label="Data feeds",
            selection_mode="multi",
            options=[
                "Finished Reporting",
                "Cables",
                "Field notes",
                "Verified News Reports",
            ],
            default=[
                "Finished Reporting",
                "Cables",
                "Field notes",
                "Verified News Reports",
            ],
            help="This does nothing right now",
        )

    doc_upload = st.sidebar.toggle("**Upload a Document to Search**")
    if doc_upload:
        file_upload = st.sidebar.file_uploader(
            "Upload document to use as conversation context", type=["pdf"]
        )
        if file_upload:
            file_upload_progress = st.sidebar.progress(
                value=0, text="File upload progress"
            )
            pdf_contents = read_pdf(file_upload, file_upload_progress)

    st.sidebar.divider()
    st.sidebar.markdown("**Connections**")
    (c1, c2) = st.sidebar.columns(2)
    with c1:
        if llm_client:
            st.success(
                "**✅ LLM**",
            )
        else:
            st.error(
                "**❌ LLM**",
            )
    with c2:
        if es:
            st.success(
                "**✅ Elastic**",
            )
        else:
            st.error(
                "**❌ Elastic**",
            )

    st.sidebar.divider()
    if st.sidebar.button("Reset Current Chat History", icon=":material/delete:"):
        st.session_state["messages"] = []

    for msg in st.session_state.messages:
        st.chat_message(name=msg["role"]).write(msg["content"])
        if "es_hits" in msg:
            st.markdown("**Cited Intelligence Reports:**")
            for x in range(len(msg["es_hits"])):
                doc = msg["es_hits"][x]
                with st.expander(
                    f"**Intelligence Report ID {doc['report_id']}** - {doc['summary'][:60]}..."
                ):
                    st.markdown(f"**Classification**: {doc['classification']}")
                    st.markdown(f"**Compartments**: {', '.join(doc['compartments'])}")
                    st.markdown(
                        f"**Report Date**: {datetime.strptime(doc['date'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%A, %B %d, %Y')}"
                    )
                    st.markdown(
                        f"**Country**: {doc['country.name'] if 'country.name' in doc else doc['country']['name']}"
                    )
                    st.markdown(f"**Source of Intel**: {doc['source']}")
                    if isinstance(doc["details"], list):
                        st.markdown(f"**Details**: {doc['details'][0]}")
                    else:
                        st.markdown(f"**Details**: {doc['details']}")

    if prompt := st.chat_input():
        with st.chat_message("user"):
            st.markdown(prompt)
        if rag:
            filters = {
                "date_range": date_range_selection,
                "classifications": classification_selection,
                "sources": source_selection,
                "countries": country_selection,
                "compartments": compartment_selection,
                "geo": get_geo_filter(prompt, llm_client, llm_model_selection),
            }
            es_hits = elasticsearch_elser(prompt, filters)["source_docs"]
            system_prompt = f"""
                Intelligence Reports:
                {str(es_hits)}

                Instructions:
                Be concise and succint.
                Answer the user's question using the provided intelligence reports text if there is enough information.
                Keep your answer grounded in the facts of the intelligence reports.
                If there isn't anything in the reports pertaining to the user's question, you may use your base foundation training knowledge to answer.
                Answer as if you are addressing a US intelligence analyst or Military officer.
                Keep in mind today's date is {today}.
                Cite the report ID in brackets that was used to generate the answer. If there was no relevant report, note that at the end.
            """
        elif doc_upload:
            es_hits = None
            system_prompt = f"""
                Document text:
                {pdf_contents}

                Instructions:
                Be concise and succint.
                Answer the user's question using the provided document text if there is enough information.
                Answer as if you are addressing a US intelligence analyst or Military officer.
                Keep in mind today's date is {today}.
                If there isn't anything in the document pertaining to the user's question, you may use your base foundation training knowledge to answer.
                Don't hallucinate.
                Don't ask follow up questions.
            """
        else:
            es_hits = None
            system_prompt = f"""
                Instructions:
                Be concise and succint.
                Don't respond back confirming this prompt.
                Answer as if you are addressing a US intelligence analyst or Military officer.
                Keep in mind today's date is {today}.
                If you don't know the answer, say that you don't know.
                Don't hallucinate.
                Don't ask follow up questions.
            """
        with st.chat_message("assistant"):
            # with st.spinner("Thinking..."):
            response = llm_client.chat.completions.create(
                model=llm_model_selection,
                messages=st.session_state.messages
                + [{"role": "user", "content": prompt}]
                + [
                    {
                        "role": "system",
                        "content": system_prompt,
                    }
                ],
                temperature=0,
            )
            if response.choices[0].finish_reason == "content_filter":
                msg = "The LLM response was filtered by the provider due to content filter. Please try a different prompt."
            else:
                msg = response.choices[0].message.content
            st.write(msg)
            if es_hits:
                asisst_resp_history_obj = {
                    "role": "assistant",
                    "content": msg,
                    "es_hits": es_hits,
                }
            else:
                asisst_resp_history_obj = {"role": "assistant", "content": msg}
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append(asisst_resp_history_obj)

        if "es_hits" in asisst_resp_history_obj:
            st.markdown("**Cited Intelligence Reports:**")
            for x in range(len(asisst_resp_history_obj["es_hits"])):
                doc = asisst_resp_history_obj["es_hits"][x]
                with st.expander(
                    f"**Intelligence Report ID {doc['report_id']}** - {doc['summary'][:64]} ..."
                ):
                    st.markdown(f"**Classification**: {doc['classification']}")
                    st.markdown(f"**Compartments**: {', '.join(doc['compartments'])}")
                    st.markdown(
                        f"**Report Date**: {datetime.strptime(doc['date'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%A, %B %d, %Y')}"
                    )
                    st.markdown(
                        f"**Country**: {doc['country.name'] if 'country.name' in doc else doc['country']['name']}"
                    )
                    st.markdown(f"**Source of Intel**: {doc['source']}")
                    if isinstance(doc["details"], list):
                        st.markdown(f"**Details**: {doc['details'][0]}")
                    else:
                        st.markdown(f"**Details**: {doc['details']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Frontend web app for RAG chatbot intel analysis demo"
    )
    parser.add_argument(
        "-c", "--config", action="store", dest="config_path", default="config.toml"
    )
    args = parser.parse_args()

    config = read_config(args.config_path)

    es = connect_es(config)

    countries = read_file("data/countries.json")
    groups = read_file("data/groups.json")
    sources = read_file("data/sources.json")
    classifications = read_file("data/classifications.json")
    compartments = read_file("data/compartments.json")

    main()
