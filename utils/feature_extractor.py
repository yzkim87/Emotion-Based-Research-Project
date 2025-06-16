import os

os.environ["TRITON_DISABLED"] = "1"

import torch
from transformers import AutoTokenizer, AutoModel
import opensmile
import tempfile
import soundfile as sf  # ✅ Tensor를 wav로 저장하기 위해 필요


class RoBERTaExtractor:
    """
    Hugging Face FacebookAI/roberta-base 사용.
    flash-attn 비활성화로 안전.
    """

    def __init__(self, model_name="FacebookAI/roberta-base"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)

    def encode(self, texts, max_length=128):
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings


# ==================================
# AUDIO: openSMILE Extractor
# ==================================


class OpenSMILEExtractor:
    def __init__(self, config="eGeMAPSv02"):
        """
        openSMILE의 eGeMAPSv02 Feature Set으로 음성 특성 추출
        """
        self.smile = opensmile.Smile(
            feature_set=getattr(opensmile.FeatureSet, config),
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def extract(self, wav_input, sampling_rate=16000):
        """
        Args:
            wav_input (str or torch.Tensor)
        Returns:
            torch.Tensor [D]
        """
        if isinstance(wav_input, str):
            feats = self.smile.process_file(wav_input)
        elif isinstance(wav_input, torch.Tensor):
            y = wav_input.cpu().numpy()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, y, sampling_rate)
                tmp_path = tmp.name
            feats = self.smile.process_file(tmp_path)
            os.remove(tmp_path)
        else:
            raise ValueError("Input must be str (path) or torch.Tensor")

        # ✅ Convert to numpy → torch.Tensor
        return torch.tensor(feats.values.squeeze(), dtype=torch.float32)
