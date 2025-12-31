"""
collector.py

Script which collects token-level logits from LLMs during zero-shot prompting.

Adapted from:
@inproceedings{trilemma2025preprint,
  title={The Trilemma of Truth in Large Language Models},
  author={Savcisens, Germans and Eliassi‐Rad, Tina},
  booktitle={arXiv preprint arXiv:2506.23921},
  year={2025}
}

"""

import torch
import logging
from collections import Counter
from abc import ABC, abstractmethod

log = logging.getLogger("logit_collector")

BINARY_TRUE = ["true", "correct", "1", "yes", "right"]
BINARY_FALSE = ["false", "incorrect", "0", "no", "wrong"]

MULTICHOICE = ["1", "2", "3", "4"]

LEGAL_AGG = ["sum", "mean", "max"]
LEGAL_QT = ["binary", "binary_true", "multichoice"]


class LogitCollectorTemplate(ABC):
    """
    Base template for collecting logits / probabilities from a model's output
    over the vocabulary, restricted to a set of tokens corresponding to
    answer options.

    Subclasses are responsible for:
      - checking the prompt template type,
      - mapping enumeration strings to token IDs,
      - and defining the actual aggregation logic in collect_logits().
    """

    def __init__(self, tokenizer, prompt_template, agg="sum"):
        """
        Constructor.

        :param tokenizer: tokenizer to use
        :param prompt_template: prompt template
        :param agg: aggregation function
        """
        assert agg in LEGAL_AGG, f"Only {LEGAL_AGG} aggregations are supported"
        self._check_prompt_template(prompt_template)

        # Variables
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.model_name = self.tokenizer.name_or_path  # type: ignore
        self.question_type = self.prompt_template.question_type
        self.aggregation = agg
        self.enum_list = self.prompt_template.enumeration

        # Tokens and IDs corresponding to the enumeration options
        self.tokens = self.augment_token_list(self.enum_list)
        self.ids = self.return_token_ids(self.tokens)

    def __call__(self, logits):
        """
        Convenience wrapper so that instances can be called directly.

        :param logits: LLM logits
        :return: LLM proba
        """
        return self.collect_proba(logits)

    def encode(self, token):
        """
        Encode a token (string) using the underlying tokenizer.

        :param token: token
        :return: tokenizer encoding
        """
        return self.tokenizer.encode(token)

    def decode(self, token_id):
        """
        Decode a single token ID using the underlying tokenizer.

        :param token_id: token id
        :return: token
        """
        return self.tokenizer.decode(token_id)

    @abstractmethod
    def _check_prompt_template(self, prompt_template):
        raise NotImplementedError()

    @abstractmethod
    def augment_token_list(self, tokens):
        """
        Given the enumeration list, build a structure that will be used
        to map each option to one or more token strings (e.g., "1" and " 1").
        """
        raise NotImplementedError()

    @abstractmethod
    def return_token_ids(self, tokens):
        """
        Convert token strings into token IDs, possibly filtering out shared IDs
        to avoid ambiguity between options.
        """
        raise NotImplementedError()

    @abstractmethod
    def collect_logits(self, logits):
        """
        Given logits over the full vocabulary (shape: [batch, vocab]),
        return per-option + "else" logits/probabilities.
        """
        raise NotImplementedError()

    def collect_proba(self, logits):
        """
        Return probabilities for each option plus an 'else' bucket, and
        sanity check that the total probability mass is preserved.

        :param logits: logits
        :return: probabilities
        """
        output = self.collect_logits(logits)
        assert (
            output.shape[1] == len(self.ids) + 1
        ), "Output shape is incorrect"
        # Sum over our collected probabilities should match the model's softmax
        # over the vocabulary (within a small tolerance).
        assert output.sum(-1).allclose(
            torch.softmax(logits, dim=-1).sum(-1), atol=0.01
        ), "Sum of logits does not match the sum of the collected logits"
        return output

    def collect_topn(self, logits, n=1):
        """
        Return the top-n tokens (decoded) from the full softmax distribution
        over the vocabulary.

        :param logits: logits
        :param n: top 'n'
        :return: top-n decoded tokens
        """
        output = torch.softmax(logits, dim=-1)
        return [self.decode(t) for t in torch.topk(output, n, dim=-1).indices]


class MultichoiceLogitCollector(LogitCollectorTemplate):
    """
    Collector for multichoice questions. Works with any PromptTemplate whose
    question_type == "multichoice" and whose enumeration field lists the
    option labels (e.g., ["1", "2", "3", "4", "5", "6"] or ["a", "b", "c"]).
    """

    def _check_prompt_template(self, prompt_template):
        """
        Check to make sure the prompt template fits.

        :param prompt_template: prompt template
        :return: None
        """
        assert (
            prompt_template.question_type == "multichoice"
        ), "Prompt template must be for multichoice question type"

    def augment_token_list(self, tokens):
        """
        For each enumeration string (e.g., "1"), create a small list of
        variants, e.g. ["1", " 1"], to accommodate tokenizers that use
        leading spaces.

        :param tokens: tokens
        :return: result dictionary
        """
        result = {}
        for t in tokens:
            _r = [t, f" {t}"]
            result[t] = _r
        return result

    def _check_token_ids(self, tokens):
        """
        Encode each token string for each option, track the token IDs, and
        return the set of IDs that appear in more than one option list
        (shared IDs). These will be dropped to avoid ambiguity.

        :param tokens: tokens
        :return: shared tokens
        """
        token_counts = Counter()
        for _tokens in tokens.values():
            _unique_tokens = set()
            for t in _tokens:
                encoded = self.encode(t)  # list[int]
                _unique_tokens.update(encoded)
            token_counts.update(_unique_tokens)

        shared_tokens = {token for token, count in token_counts.items() if count > 1}
        return shared_tokens

    def return_token_ids(self, tokens):
        """
        Convert token strings to token IDs, dropping shared IDs so that each
        option only uses IDs unique to that option.

        :param tokens: tokens
        :return: token ids
        """
        shared_tokens = self._check_token_ids(tokens)  # type: ignore
        output = {}
        for name, token_variants in tokens.items():
            _temp = []
            for t in token_variants:
                encoded_tokens = self.encode(t)
                # Keep only tokens that are not shared
                _temp.extend(
                    [tok for tok in encoded_tokens if tok not in shared_tokens]
                )
            # Remove duplicates within the list
            output[name] = list(set(_temp))
        return output

    def collect_logits(self, logits):
        """
        Collect probabilities for each option and an 'else' bucket.

        :param logits: logits
        :return: probabilities
        """
        assert logits.dim() == 2, "Logits must be 2D tensor"
        logits = torch.softmax(logits, dim=-1)
        output = []

        if self.aggregation == "sum":
            _else = logits.sum(-1)  # total mass, shape [batch]
            for v in self.ids.values():
                _logit = 0.0
                for vv in v:
                    _logit += logits[:, vv]
                _else -= _logit
                output.append(_logit)
            output.append(_else)
            output_tensor = torch.vstack(output).T
            assert (
                output_tensor.shape[1] == len(self.ids) + 1
            ), "Output shape is incorrect"
            return output_tensor

        raise NotImplementedError(
            f"Aggregation '{self.aggregation}' not implemented for MultichoiceLogitCollector"
        )

    def collect_logits_unsafe(self, logits):
        """
        Collect logits for NNSight without any checks (no shape / sum checks).

        :param logits: logits
        :return: output
        """
        logits = torch.softmax(logits, dim=-1)
        output = []
        if self.aggregation == "sum":
            _else = logits.sum(-1)
            for v in self.ids.values():
                _logit = 0.0
                for vv in v:
                    _logit += logits[:, vv]
                _else -= _logit
                output.append(_logit)
            output.append(_else)
            output = torch.vstack(output).T
            return output
        raise NotImplementedError(
            f"Aggregation '{self.aggregation}' not implemented for MultichoiceLogitCollector"
        )

    def collect_proba_and_argmax(self, logits):
        """
        Returns:
          - probs: tensor of shape [batch, num_options + 1]
                   (same as collect_proba)
          - argmax_idx: tensor of shape [batch], the argmax index over
                        only the labeled options (ignoring the 'else' column).

        For example, with enumeration ["a", "b", "c"]:
          argmax_idx == 0 to "a" (True)
          argmax_idx == 1 to "b" (False)
          argmax_idx == 2 to "c" (Neither)

        :param logits: logits
        :return: probabilities and idx of choice
        """
        probs = self.collect_proba(logits)  # (batch, num_options + 1)
        choice_probs = probs[:, :-1]       # drop 'else'
        choice_idx = torch.argmax(choice_probs, dim=-1)
        return probs, choice_idx
