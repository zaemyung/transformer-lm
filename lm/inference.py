import json
from pathlib import Path
from typing import List, Tuple

import sentencepiece as spm
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import fire
from .fire_utils import only_allow_defined_args

from .model import Model, HParams
from .common import END_OF_LINE, END_OF_TEXT


def yield_n_sized_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class ModelWrapper:
    END_OF_LINE = END_OF_LINE
    END_OF_TEXT = END_OF_TEXT

    def __init__(self, model: Model, sp_model: spm.SentencePieceProcessor):
        self.model = model
        self.sp_model = sp_model

    @classmethod
    def load(cls, root: Path):
        sp_model = spm.SentencePieceProcessor()
        sp_model.load(str(root / 'sp.model'))
        hparams = json.loads((root / 'params.json').read_text())['hparams']
        hparams.setdefault('n_hidden', hparams['n_embed'])
        model = Model(HParams(**hparams))
        state = torch.load(root / 'model.pt', map_location='cpu')
        state_dict = fixed_state_dict(state['state_dict'])
        model.load_state_dict(state_dict)

        tensor_list = list(state_dict.items())
        for layer_tensor_name, tensor in tensor_list:
            print("Layer %-42s: %9d elements" % (layer_tensor_name, torch.numel(tensor)))
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("Total # params: %d" % pytorch_total_params)
        return cls(model.cuda(), sp_model)

    def tokenize(self, s: str) -> List[str]:
        return self.sp_model.EncodeAsPieces(s)

    def token_to_id(self, token: str) -> int:
        return self.sp_model.PieceToId(token)

    def id_to_token(self, token_id: int) -> str:
        return self.sp_model.IdToPiece(int(token_id))

    def get_lm_scores(self, sentences: List[str], batch_size: int = 70) -> torch.Tensor:
        if isinstance(sentences, str):
            sentences = [sentences]
        sent_chunks = list(yield_n_sized_chunks(sentences, batch_size))
        results = []
        for chunk in tqdm(sent_chunks, total=len(sent_chunks)):
            res = self._get_lm_scores(chunk)
            results.extend(res)
        return results

    def _get_lm_scores(self, sentences: List[str]) -> torch.Tensor:
        with torch.no_grad():
            sentences_tok = [[self.END_OF_TEXT] + self.tokenize(s) + [self.END_OF_TEXT] for s in sentences]
            sentences_tok_len = [len(toks) - 1 for toks in sentences_tok]
            sentences_ids = [torch.LongTensor([self.token_to_id(t) for t in tokens]) for tokens in sentences_tok]

            # pad sequence
            sentences_ids_padded = pad_sequence(sentences_ids, batch_first=True).cuda()

            log_probs = torch.log_softmax(self.model(sentences_ids_padded)['logits'], dim=2)
            results = []
            for i, tokens in enumerate(sentences_tok):
                tok_len = sentences_tok_len[i]
                tok_ids = [self.token_to_id(token) for token in tokens[1:tok_len]]
                logits = torch.diag(log_probs[i, :tok_len - 1, tok_ids], 0)
                sum_prob = torch.sum(logits) / (tok_len - 1)
                results.append(sum_prob.detach().cpu().item())
        return results

    def get_log_probs(self, tokens: List[str]) -> torch.Tensor:
        """ Return a tensor with shape (len(tokens), len(self.sp_model)),
        with log-probabilities for tokens after each token in tokens.
        If this is a start of the text, you may want to prepend END_OF_TEXT:
        model.get_log_probs([model.END_OF_TEXT] + tokens).
        Use model.tokenize to obtain tokens.
        """
        assert len(tokens) <= self.model.hparams.n_ctx  # TODO
        ids = [self.token_to_id(t) for t in tokens]
        ctx = torch.LongTensor(ids).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(ctx)['logits'].squeeze(0)
            return torch.log_softmax(logits, dim=1)

    def get_occurred_log_probs(
            self, tokens: List[str]) -> List[Tuple[float, str]]:
        """ Return a list of log probs of actually occurred tokens,
        starting from the second.
        """
        log_probs = self.get_log_probs(tokens)
        out = []
        for idx, token in enumerate(tokens[1:]):
            out.append((float(log_probs[idx, self.token_to_id(token)]), token))
        return out

    def get_next_top_k(
            self, tokens: List[str], top_k: int) -> List[Tuple[float, str]]:
        """ Return a list of top k tuples of log prob and token,
        for what would come after the last token.
        """
        next_log_probs = self.get_log_probs(tokens)[-1]
        return sorted([(float(next_log_probs[i]), self.id_to_token(i))
                       for i in next_log_probs.argsort()[-top_k:]],
                      reverse=True)

    def generate_tokens(self, tokens_prefix: List[str], tokens_to_generate: int, top_k: int) -> List[str]:

        tokens = list(tokens_prefix)

        for i in range(tokens_to_generate):

            # generate TOP_K potential next tokens
            ntk = self.get_next_top_k(tokens, top_k)

            # convert log probs to real probs
            logprobs = np.array(list(map(lambda a: a[0], ntk)))
            probs = np.exp(logprobs) / np.exp(logprobs).sum()

            # pick next token randomly according to probs distribution
            next_token_n = np.random.choice(top_k, p=probs)
            next_token = ntk[next_token_n][1]
            # print (next_token)

            tokens.append(next_token)

        return tokens


def fixed_state_dict(state_dict):
    if all(k.startswith('module.') for k in state_dict):
        # legacy multi-GPU format
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict


def gen_main(model_path, prefix, tokens_to_generate=42, top_k=8):

    print("loading model from %s" % model_path)
    mw = ModelWrapper.load(Path(model_path))

    print("generating text for prefix %s" % prefix)
    tokens = mw.tokenize(prefix)

    tokens_gen = mw.generate_tokens(tokens, tokens_to_generate, top_k)
    print(mw.sp_model.DecodePieces(tokens_gen))


def fire_gen_main():
    fire.Fire(only_allow_defined_args(gen_main))
