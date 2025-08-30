1. Install dependencies: `pip install torch transformers sentencepiece gradio`

- `torch`: Engine to execute model
- `transformers`: Model loader through pipelines 
- `sentencepiece`: Tokenizers to convert text into numbers to feed into models 
- `gradio`: Quickly build a demo or web application for your machine learning model

Known issues:

1. `.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See:` https://github.com/urllib3/urllib3/issues/3020
- Might need to downgrade urllib3:  `pip install "urllib3<2"`
