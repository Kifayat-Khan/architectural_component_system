# architectural_component_system

This system is developed to quickly document and analyze Taichung’s historic buildings from photos. It helps preserve cultural heritage by turning images into measurable features, factual narratives, and shareable reports. The goal is to support research, urban planning, and public awareness with a consistent, fast, and reliable method.

## OpenAI API Key Setup (Local)

This project reads the OpenAI API key from Streamlit secrets.

1) Create the file:
   `.streamlit/secrets.toml`

2) Add your key (do not add extra spaces or quotes):
   ```toml
   OPENAI_API_KEY="YOUR_OPENAI_API_KEY"


# for installion 
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py



streamlit run app.py