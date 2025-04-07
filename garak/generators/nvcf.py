# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVCF LLM interface"""

import json
import logging
import time
from typing import List, Union
import requests, io
import base64
import io
from io import BytesIO
from PIL import Image as _PILImage
# from vlm_client import VLMClient

import backoff
import requests

from garak import _config
from garak.exception import ModelNameMissingError, BadGeneratorException
from garak.generators.base import Generator


## GLOBALS 
NVCF_API_TOKEN = "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"
NVCF_FUNCTION_NAME="85638347-c3ff-4eda-a8e5-1a03fb2dfd1c"
NVCF_MODEL_NAME="cosmos-nemotron-4B"

def base64_image(img):
    """
    Resize an image encoded as a Base64 string.
    Args:
    base64_string (str): Base64 string of the original image.
    size (tuple): Desired size of the image as (width, height).
    Returns:
    str: Base64 string of the resized image.
    """
    buffered = io.BytesIO()
    img.save(buffered, format=img.format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def ping_server(prompt, payload, url, headers):
    global NVCF_API_TOKEN
    global NVCF_FUNCTION_NAME
    global NVCF_MODEL_NAME

    url = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/" + NVCF_FUNCTION_NAME
    token=NVCF_API_TOKEN


    headers = {
        "Authorization": f"Bearer {token}",
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    text = prompt["text"]

    text = text.replace('<image>', '')
    # print(' text prompt : ')
    # print(text)
    
    # print("####")
    image_filename = prompt["image"]
    # print(image_filename)    
    payload = {
    "model": NVCF_MODEL_NAME,
    "temperature": 1.0,
    "top_p": 0.0,  
    "top_k": 1.0,  
    "max_tokens": 512,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image(_PILImage.open(image_filename))}"
                    }
                }
            ]
        }
    ]
    }
    response = requests.post(url, headers=headers, json=payload)
    # print('Status code ')
    # print(response.status_code)
    # assert response.status_code == 500
    return response

class NvcfChat(Generator):
    """Wrapper for NVIDIA Cloud Functions Chat models via NGC. Expects NVCF_API_KEY environment variable."""

    # ENV_VAR = "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"
    ENV_VAR = "NVCF_API_KEY"
    DEFAULT_PARAMS = Generator.DEFAULT_PARAMS | {
        "temperature": 0.2,
        "top_p": 0.7,
        "status_uri_base": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/",
        "invoke_uri_base": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/",
        "timeout": 60,
        "version_id": None,  # string
        "stop_on_404": True,
        "extra_params": {  # extra params for the payload, e.g. "n":1 or "model":"google/gemma2b"
            "stream": False
        },
    }

    supports_multiple_generations = False
    generator_family_name = "NVCF"

    def __init__(self, name=None, config_root=_config):
        self.name = name
        self._load_config(config_root)
        self.local_mode = True
        # self.vlm_client = VLMClient()
        print('VLM model loaded ..')
        # print(' ## Loaded COnfig ##')
        self.fullname = (
            f"{self.generator_family_name} {self.__class__.__name__} {self.name}"
        )
        self.seed = _config.run.seed
        # self.seed = 9999

        if self.name is None:
            raise ModelNameMissingError(
                "Please specify a function identifier in model name (-n)"
            )

        self.invoke_uri = self.invoke_uri_base + self.name

        if self.version_id is not None:
            self.invoke_uri += f"/versions/{self.version_id}"

        super().__init__(self.name, config_root=config_root)

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        # print(' NVCF CHAT constructor end ## ')

    def _prepare_prompt(self, prompt):
        # print(' enter parent prepare prompt')
        return prompt

    def _build_payload(self, prompt) -> dict:
        # print('enter parent build payload')
        prompt = self._prepare_prompt(prompt)

        payload = {
            "messages": [{"content": prompt, "role": "user"}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        for k, v in self.extra_params.items():
            payload[k] = v

        return payload

    def _extract_text_output(self, response) -> str:
        return [c["message"]["content"] for c in response["choices"]]

    @backoff.on_exception(
        backoff.fibo,
        (
            AttributeError,
            TimeoutError,
            requests.exceptions.HTTPError,
            requests.exceptions.ConnectionError,
        ),
        max_value=70,
    )
    def _call_model(
        self, prompt: str | List[dict], generations_this_call: int = 1
    ) -> List[Union[str, None]]:
        # print(" ### Enter NVCF Chat call model () ## ")
        session = requests.Session()

        payload = self._build_payload(prompt)

        # print(' Generated Payload ##')
        # print(payload)

        ## NB config indexing scheme to be deprecated
        # config_class = f"nvcf.{self.__class__.__name__}"
        # if config_class in _config.plugins.generators:
        #     if "payload" in _config.plugins.generators[config_class]:
        #         for k, v in _config.plugins.generators[config_class]["payload"].items():
        #             payload[k] = v

        if self.seed is not None:
            payload["seed"] = self.seed
        # print(payload.keys())

        request_time = time.time()
        logging.debug("nvcf : payload %s", repr(payload))
        # print(' BEfore post request ')
        # print(self.invoke_uri)
        # print(self.headers)
        prompt['text'] = prompt['text'].replace('<image>', '')

        # print(prompt)
        if self.local_mode:
                response = self.vlm_client.ping_client(text = prompt['text'], images = [prompt['image']])
                return [response]
    # print(response)
        response = ping_server(prompt, payload, self.invoke_uri, self.headers)
        # print(payload)
        # response = session.post(self.invoke_uri, headers=self.headers, json=payload)
        # print("## response code ## ")
        # print(response.status_code)
        if response.status_code == 500:
            logging.debug(response.json())
        while response.status_code == 202:
            if time.time() > request_time + self.timeout:
                raise TimeoutError("NVCF Request timed out")
            request_id = response.headers.get("NVCF-REQID")
            if request_id is None:
                msg = "Got HTTP 202 but no NVCF-REQID was returned"
                logging.info("nvcf : %s", msg)
                raise AttributeError(msg)
            status_uri = self.status_uri_base + request_id
            response = session.get(status_uri, headers=self.headers)

        if 400 <= response.status_code < 600:
            logging.warning("nvcf : returned error code %s", response.status_code)
            logging.warning("nvcf : returned error body %s", response.content)
            if response.status_code == 400 and prompt == "":
                # error messages for refusing a blank prompt are fragile and include multi-level wrapped JSON, so this catch is a little broad
                return [None]
            if response.status_code == 404 and self.stop_on_404:
                msg = "nvcf : got 404, endpoint unavailable, stopping"
                logging.critical(msg)
                print("\n\n" + msg)
                print("nvcf :", response.content)
                raise BadGeneratorException()
            if response.status_code >= 500:
                if response.status_code == 500 and json.loads(response.content)[
                    "detail"
                ].startswith("Input value error"):
                    logging.warning("nvcf : skipping this prompt")
                    return [None]
                else:
                    response.raise_for_status()
            else:
                logging.warning("nvcf : skipping this prompt")
                return [None]

        else:
            response_body = response.json()
            # print(' Received response')
            return self._extract_text_output(response_body)


class VLMChat(NvcfChat):
    """ Class for VLM model text + image -> text """

    # DEFAULT_PARAMS =  {
    #     "suppressed_params": {"n", "frequency_penalty", "presence_penalty", "stop"},
    #     "max_image_len": 180_000,
    # }

    modality = {"in": {"text", "image"}, "out": {"text"}}

    def __init__(self, nvcf_function_name = None, model_name = None, config_root=_config):
        # print(' ## Enter VLMChat init () ')
        model_name = "Nemovision-4B-v2-instruct"
        model_name =  "cosmos-nemotron-4B"
        self.model_name = model_name
        nvcf_function_name = "69053ae4-f69e-412f-9f83-c36de1b3ff17"
        nvcf_function_name = "85638347-c3ff-4eda-a8e5-1a03fb2dfd1c"
        super().__init__(name = nvcf_function_name, config_root = _config) 

        self.local_mode = False
         # Call the parent class constructor
        # print(' ## Exit VLMChat init () ')
        # self.age = age

    def _prepare_prompt(self, prompt):
        # print(" #nter prepare prompt ")
        if isinstance(prompt, str):
            prompt = {"text": prompt, "image": None}

        text = prompt["text"]
        image_filename = prompt["image"]
        # print(image_filename)
        # if self.local_mode:
        #     processed_prompt = {"text": text, "image": image_filename}
        # else:
        #     processed_prompt = {"text": text, "image": base64_image(_PILImage.open(image_filename))}
        processed_prompt = {"text": text, "image": image_filename}
        return processed_prompt
        pass ## prep text + image input here 

    def _build_payload(self, prompt) -> dict:
        # print('enter build payload')
        prompt = self._prepare_prompt(prompt)

        # payload = {
        # "messages": [{"content": prompt, "role": "user"}],
        # "temperature": self.temperature,
        # "top_p": self.top_p,
        # "max_tokens": self.max_tokens,
        # "stream": False,
        # }

        # for k, v in self.extra_params.items():
        #     payload[k] = v
        # print(prompt.keys())
        # print(prompt['text'])
        # print(' #Enter VLMChat payload build ## ')
        if self.local_mode:
            payload = prompt
        else:
            payload = {
                "model": self.model_name,
                "temperature": 1.0,
                "top_p": 0.0,  
                "top_k": 1.0,  
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text":  prompt['text']},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{prompt['image']}"
                                }
                            }
                        ]
                    }
                ]
            }
        # print(' Exit payload build ##')
        return payload
        pass # prep pakcet with inference params 
        # print("Response:", response.json()['choices'][0]['message']['content'][0]['text'])

    def _extract_text_output(self, response) -> str:
        # print(' Extracted output : ')
        # print(response['choices'][0]['message']['content'])
        return [response['choices'][0]['message']['content']]

class NvcfCompletion(NvcfChat):
    """Wrapper for NVIDIA Cloud Functions Completion models via NGC. Expects NVCF_API_KEY environment variables."""

    def _build_payload(self, prompt) -> dict:

        payload = {
            "prompt": prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        for k, v in self.extra_params.items():
            payload[k] = v

        return payload

    def _extract_text_output(self, response) -> str:
        return [c["text"] for c in response["choices"]]


DEFAULT_CLASS = "NvcfChat"
