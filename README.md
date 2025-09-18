## Virtual environment with .Venv

1. Create .venv: `python3.10 -m venv .venv` this will be created in the root folder
2. Activate .venv: `source .venv/bin/activate` in your terminal it should show .venv
3. Update pip: `python -m pip install --upgrade pip` to install, upgrade, and manage libraries.
4. Install dependencies: `pip install -r requirements.txt` or `pip install torch transformers sentencepiece gradio` (without specifying a file)

Ref.: https://gist.github.com/GinoAvanzini/f0ed9c1a74ffce3f832c9fa68f19daba

## Dependencies

- `torch`: Engine to execute model
- `transformers`: Model loader through pipelines stored in `~/.cache/huggingface/hub`
- `huggingface_hub`: Download models (installed along with `transformers` by default. Run `huggingface-cli login` to set access token stored in `~/.cache/huggingface/token`. To confirm login `hf auth whoami`
- `sentencepiece`: Tokenizers to convert text into numbers to feed into models 
- `gradio`: Quickly build a demo or web application for your machine learning model

## Create .env

1. Create a file `.env` in the root directory:

```
HF_TOKEN=ACCESS_TOKEN # https://huggingface.co/settings/tokens
```
 

## Known issues:

1. `.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See:` https://github.com/urllib3/urllib3/issues/3020: Might need to downgrade urllib3:  `pip install "urllib3<2"`

## References:

- https://medium.com/@vignarajj/how-to-use-hugging-face-effectively-as-a-mobile-developer-a-practical-guide-with-real-world-02ac3a0d688b
