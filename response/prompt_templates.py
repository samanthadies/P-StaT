"""
prompt_templates.py

Prompt templates for zero-shot experiments.

Adapted from:
@inproceedings{trilemma2025preprint,
  title={The Trilemma of Truth in Large Language Models},
  author={Savcisens, Germans and Eliassi‐Rad, Tina},
  booktitle={arXiv preprint arXiv:2506.23921},
  year={2025}
}

"""

import logging
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)

LEGAL_TASKS = ["binary", "multichoice"]
LEGAL_TYPES = ["default", "masked", "instruct"]


class PromptTemplate(ABC):
    """
    Base template for formatting prompts given a statement and a set of
    answer options. Subclasses implement the specific answer texts and
    question type.
    """

    def __init__(self, task=0, prompt_type="default", enumeration=None, system_role="system",
                 user_role="user", assist_role="assistant"):
        """
        Constructor.

        :param task: task
        :param prompt_type: type of prompt
        :param enumeration: enumarated options for LLM response
        :param system_role: system role (instruct templates)
        :param user_role: user role (instruct templates)
        :param assist_role: assist role (instruct templates)
        """
        assert prompt_type in LEGAL_TYPES, f"Prompt type must be one of {LEGAL_TYPES}"
        self.task = task
        self.prompt_type = prompt_type
        self.enumeration = self.format_enumeration_list(enumeration or [])
        # Used only for the instruct template
        self.system_role = system_role
        self.user_role = user_role
        self.assist_role = assist_role

    def format_enumeration_list(self, enumeration):
        """
        Format the enumeration of answer options.

        :param enumeration: enumerations
        :return: list
        """
        return [str(e) for e in enumeration]

    def __call__(self, statement):
        """
        Returns the prompt.

        :param statement: statement
        :return: prompt
        """
        return self.get_prompt(statement)

    @property
    @abstractmethod
    def enum_unit(self):
        raise NotImplementedError()

    @property
    def templates(self):
        """
        Default templates that prepend a generic “Is the following statement
        correct?” text. Subclasses like ABC3Prompt can override this to change
        the structure entirely.

        :return: default prompt template
        """
        return {
            "default": lambda x: (
                f"Question: Is the following statement correct? {x}\n\n"
                "Select one of the following options:\n"
                f"{self.options_as_str()}\n\n"
                f"Please respond with the corresponding {self.enum_unit}. "
                f"The final answer is {self.enum_unit} "
            ),
            "masked": lambda x: (
                f"Question: Is the following statement correct? {x}\n\n"
                "Select one of the following options:\n"
                f"{self.options_as_str()}\n\n"
                f"Please respond with the corresponding {self.enum_unit}. "
                f"The final answer is {self.enum_unit} [MASK]."
            ),
            "instruct": lambda x: self.instruct_template(x),
        }

    def return_intro(self):
        """
        Utility to extract just the “intro” part of the prompt (before the
        baseline statement) for debugging.

        :return: intro of the prompt
        """
        baseline = "Three plus three equals six."
        if self.prompt_type == "default":
            prompt = self.get_prompt(baseline)
            return prompt.split(baseline)[0]
        elif self.prompt_type == "instruct":
            prompt = self.get_prompt(baseline)
            prompt_baseline: list[dict] = []
            for i in range(len(prompt)):
                prompt_baseline.append(prompt[i])
                if baseline in prompt[i]["content"]:
                    prompt_baseline[i]["content"] = prompt[i]["content"].split(
                        baseline
                    )[0]
                    break
            return prompt_baseline

    def options_as_dict(self):
        """
        Return the enumerated answer options as a dictionary.

        :return: dictionary of enumerated options
        """
        return {str(e): a for e, a in zip(self.enumeration, self.answers)}

    def options_as_str(self):
        """
        List the enumerated answer options as a string.

        :return: string of enumerated options
        """
        return "\n".join([f"{e}. {a}" for e, a in self.options_as_dict().items()])

    @property
    @abstractmethod
    def question_type(self):
        raise NotImplementedError()

    def get_prompt(self, statement):
        """
        Map a raw statement to a formatted prompt. For default templates, this is
        a single string. For instruct templates, it returns a list of chat
        messages (dicts with 'role' and 'content').

        :param statement: statement
        :return: prompt
        """
        assert isinstance(statement, str), "Statement must be a string."
        assert len(statement) > 2, (
            "Statement is too short. Please provide a longer statement."
        )
        return self.templates[self.prompt_type](statement)

    def instruct_template(self, statement):
        """
        Default instruct-style template: wraps the statement in a system/user/
        assistant conversation.

        :param statement: statement
        :return: instruct prompt
        """
        prompt = (
            f"Question: Is the following statement correct? {statement}\n\n "
            f"Select one of the following options:\n{self.options_as_str()}\n"
        )
        if self.system_role == self.user_role:
            return [
                {
                    "role": f"{self.user_role}",
                    "content": (
                        "You are an expert in fact-checking. Your task is to assist the user by answering questions "
                        f"based on your comprehensive knowledge. Please respond with the corresponding {self.enum_unit}.\n\n"
                        f"{prompt}"
                    ),
                },
                {
                    "role": f"{self.assist_role}",
                    "content": f"The final answer is {self.enum_unit} ",
                },
            ]
        else:
            return [
                {
                    "role": f"{self.system_role}",
                    "content": (
                        "You are an expert in fact-checking. Your task is to assist the user by answering questions based "
                        f"on your comprehensive knowledge. Please respond with the corresponding {self.enum_unit}."
                    ),
                },
                {"role": f"{self.user_role}", "content": f"{prompt}"},
                {
                    "role": f"{self.assist_role}",
                    "content": f"The final answer is {self.enum_unit} ",
                },
            ]

    @property
    @abstractmethod
    def answers(self):
        raise NotImplementedError()

    @property
    def num_answers(self):
        return len(self.answers)

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, task):
        """
        Set the task.

        :param task: task
        :return: None
        """
        assert task in [
            0,
            1,
        ], "Task must be 0 for False -> Truth or 1 for Null->Know"
        self._task = task

    @property
    def enumeration(self):
        """
        Set the enumeration.

        :return: enumeration
        """
        return self._enumeration

    @enumeration.setter
    def enumeration(self, enumeration):
        """
        Set the enumeration.

        :return: enumeration
        """
        assert len(enumeration) == len(
            self.answers
        ), "The number of options must match the number of answers."
        self._enumeration = enumeration


class BinaryPrompt(PromptTemplate):
    """
    Prompt template with binary answers.
    """

    def __init__(
        self, task=0, prompt_type="default", enumeration=["1", "2"], system_role="system",
            user_role="user", assist_role="assistant"):
        """
        Constructor.

        :param task: task
        :param prompt_type: type of prompt
        :param enumeration: enumarated options for LLM response
        :param system_role: system role (instruct templates)
        :param user_role: user role (instruct templates)
        :param assist_role: assist role (instruct templates)
        """
        super().__init__(
            task=task,
            prompt_type=prompt_type,
            enumeration=enumeration,
            system_role=system_role,
            user_role=user_role,
            assist_role=assist_role,
        )

    @property
    def question_type(self):
        return "binary"

    @property
    def enum_unit(self):
        return "number"

    @property
    def answers(self):
        return ["Yes", "No"]


class MultichoicePrompt(PromptTemplate):
    """
    Prompt template with multiple-choice answers.
    """

    def __init__( self, task=0, prompt_type="default", enumeration=["1", "2", "3", "4", "5", "6"],
                  system_role="system", user_role="user", assist_role="assistant"):
        """
        Constructor.

        :param task: task
        :param prompt_type: type of prompt
        :param enumeration: enumarated options for LLM response
        :param system_role: system role (instruct templates)
        :param user_role: user role (instruct templates)
        :param assist_role: assist role (instruct templates)
        """
        super().__init__(
            task=task,
            prompt_type=prompt_type,
            enumeration=enumeration,
            system_role=system_role,
            user_role=user_role,
            assist_role=assist_role,
        )

    @property
    def question_type(self):
        return "multichoice"

    @property
    def enum_unit(self):
        return "number"

    @property
    def answers(self):
        return [
            "The statement is correct",
            "The statement is incorrect",
            "I do not have sufficient knowledge about the statement",
            "The statement is too ambiguous to provide a reliable answer",
            "All of the above options are correct",
            "None of the above options are applicable",
        ]


class MultichoicePromptTF(MultichoicePrompt):
    @property
    def answers(self):
        return [
            "The statement is true",
            "The statement is false",
            "I do not have sufficient knowledge about the statement",
            "The statement is too ambiguous to provide a reliable answer",
            "All of the above options are correct",
            "None of the above options are applicable",
        ]


class MultichoicePromptABC(MultichoicePrompt):
    """
    Multiple choice prompt with alphabetic responses.
    """

    def __init__(self, task=0, prompt_type="default", enumeration=["A", "B", "C", "D", "E", "F"],
                 system_role="system", user_role="user", assist_role="assistant"):
        """
        Constructor.

        :param task: task
        :param prompt_type: type of prompt
        :param enumeration: enumarated options for LLM response
        :param system_role: system role (instruct templates)
        :param user_role: user role (instruct templates)
        :param assist_role: assist role (instruct templates)
        """

        super().__init__(
            task=task,
            prompt_type=prompt_type,
            enumeration=enumeration,
            system_role=system_role,
            user_role=user_role,
            assist_role=assist_role,
        )

    @property
    def enum_unit(self):
        return "the uppercase letter"


class ABC3Prompt(PromptTemplate):
    """
    Only ABC prompt template
    """

    def __init__(self, task=0, prompt_type="default", enumeration=None, system_role="system",
                 user_role="user",assist_role="assistant"):
        """
        Constructor.

        :param task: task
        :param prompt_type: type of prompt
        :param enumeration: enumarated options for LLM response
        :param system_role: system role (instruct templates)
        :param user_role: user role (instruct templates)
        :param assist_role: assist role (instruct templates)
        """
        if enumeration is None:
            enumeration = ["a", "b", "c"]
        super().__init__(
            task=task,
            prompt_type=prompt_type,
            enumeration=enumeration,
            system_role=system_role,
            user_role=user_role,
            assist_role=assist_role,
        )

    @property
    def question_type(self):
        return "multichoice"

    @property
    def enum_unit(self):
        return "letter"

    @property
    def answers(self):
        # Must match enumeration length; these are exactly the option texts.
        return [
            "The statement is true.",
            "The statement is false.",
            "The statement is neither true nor false.",
        ]

    @property
    def templates(self):
        """
        Override the default templates to add the choice list and final answer instruction.

        :return: template
        """
        return {
            "default": lambda x: (
                f"{x}\n"
                "Select one of the following options:\n"
                f"{self.options_as_str()}\n"
                "Please respond with the corresponding letter. The final answer is "
            ),
            "masked": lambda x: (
                f"{x}\n"
                "Select one of the following options:\n"
                f"{self.options_as_str()}\n"
                "Please respond with the corresponding letter. The final answer is [MASK]."
            ),
            "instruct": lambda x: self.instruct_template(x),
        }

    def instruct_template(self, statement):
        """
        Chat-style version of the same prompt.

        :param statement: statement
        :return: chat template
        """

        prompt = (
            f"{statement}\n"
            "Select one of the following options:\n"
            f"{self.options_as_str()}\n"
        )

        if self.system_role == self.user_role:
            return [
                {
                    "role": f"{self.user_role}",
                    "content": (
                        "You are an expert in fact-checking. Your task is to assist the user "
                        "by answering questions based on your comprehensive knowledge. "
                        "Please respond with the corresponding letter.\n\n"
                        f"{prompt}"
                    ),
                },
                {
                    "role": f"{self.assist_role}",
                    "content": "The final answer is ",
                },
            ]
        else:
            return [
                {
                    "role": f"{self.system_role}",
                    "content": (
                        "You are an expert in fact-checking. Your task is to assist the user "
                        "by answering questions based on your comprehensive knowledge. "
                        "Please respond with the corresponding letter."
                    ),
                },
                {"role": f"{self.user_role}", "content": prompt},
                {
                    "role": f"{self.assist_role}",
                    "content": "The final answer is ",
                },
            ]
