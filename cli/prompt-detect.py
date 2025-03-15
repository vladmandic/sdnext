# Example:
# > python cli/lang-detect.py "have a good day"
# > ['eng_latn:1.00']
# eng=language, latn=latin alphabet, 1.00=confidence

import sys
import fasttext
from huggingface_hub import hf_hub_download


repo_id = "facebook/fasttext-language-identification"
model = None


def detect(text:str, top:int=1, threshold:float=0.25) -> str:
    try:
        global model # pylint: disable=global-statement
        if model is None:
            model_path = hf_hub_download(repo_id, filename="model.bin")
            model = fasttext.load_model(model_path)
        lang, score = model.predict(text, k=top, threshold=threshold, on_unicode_error="ignore")
        result = [f"{l.replace("__label__", "").lower()}:{s:.2f}" for l, s in zip(lang, score) if s > threshold][:top]
        return result
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <text>")
    else:
        print(detect(sys.argv[1]))
