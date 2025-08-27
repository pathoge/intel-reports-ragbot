## GenAI-Powered Intelligence Analysis Demo

This repo contains artifacts & Python programs to generate completely fake intelligence reports, index them into Elasticsearch, and query them using a Retrieval Augmented Generation (RAG) demo app.

![screenshot](screenshot.png)

### Quickstart - Docker (recommended)
0. Prereqs:
    - Install Docker
    - Create an Elastic Cloud deployment. You'll need the cloud ID and an API key for it.
        - Download and deploy the ELSER v2 model with the name `.elser_model_2_linux-x86_64`.
    - Get an Azure OpenAI deployment + credentials.

1. Clone this repo & change directories into the repo folder

2. Build the Docker container:
```
docker build --no-cache -t ia-genai-demo:latest .
```

3. While that's running, make a copy of `config.toml`, rename it to something memorable, and fill out the required settings (note - this app is designed to be used by Elasticians with our internal LLM proxy):
```
NUM_REPORTS = number of fake reports to generate

ELASTIC_CLOUD_ID = Elastic cloud ID to send output to
ELASTIC_API_KEY = Elasticseach API key to use (optional if using username/password)
ELASTIC_USER = Elasticsearch username to use (optional if using api key)
ELASTIC_PASSWORD = Elasticsearch password to use (optional if using api key)
ELASTIC_INDEX = "intel-reports"

LLM_PROXY_BASE_URL = LLM proxy URL
LLM_PROXY_API_KEY = LLM proxy API key
```

4. Run the container and pass in your `config.toml` at runtime:
```
docker run --rm -v ./config-pathoge.toml:/app/config.toml --name ia-genai-demo -p 8501:8501 ia-genai-demo:latest
```
NOTE: On Windows you may have to run a slightly modified command:
```
docker run --rm -v //$(PWD)/config-pathoge.toml:/app/config.toml --name ia-genai-demo -p 8501:8501 ia-genai-demo:latest
```

5. Navigate to http://localhost:8501 in your favorite web browser.

6. (Initial run) In the sidebar of the web app, check the "Data setup" box and click the button to generate the intelligence reports and send them to your Elasticsearch cluster. WARNING: this action deletes the index if it already exists. It also requires the Elasticsearch cluster to have the ELSER v2 model already deployed and running with the name `.elser_model_2_linux-x86_64`. 

### Quickstart - Bare Python
1. Clone this repo

2. Ensure dependencies are installed:
```
pip3 install -r requirements.txt
```

3. See step 3 above. 

4. To start the application, run the following command and substitute in your custom `config.toml`.
```
streamlit run genai-intel-demo.py -- --config config-pathoge.toml
```

5. See step 5 above.

6. See step 6 above.
