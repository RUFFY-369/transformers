# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil
import tempfile
import unittest

import numpy as np
import pytest

from transformers import CLIPTokenizer, CLIPTokenizerFast, ImageBindFeatureExtractor
from transformers.testing_utils import require_torch, require_torchaudio, require_vision
from transformers.utils import is_vision_available


if is_vision_available():
    from PIL import Image

    from transformers import ImageBindImageProcessor, ImageBindProcessor

from ...test_processing_common import ProcessorTesterMixin

@require_vision
@require_torchaudio
class ImageBindProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = ImageBindProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        self.checkpoint = "EduardoPacheco/imagebind-huge"

        image_processor = ImageBindImageProcessor()
        tokenizer_slow = CLIPTokenizer.from_pretrained(self.checkpoint)
        tokenizer_fast = CLIPTokenizerFast.from_pretrained(self.checkpoint)
        feature_extractor = ImageBindFeatureExtractor()

        processor_slow = ImageBindProcessor(image_processor, tokenizer_slow, feature_extractor)
        processor_fast = ImageBindProcessor(image_processor, tokenizer_fast, feature_extractor)

        processor_slow.save_pretrained(self.tmpdirname)
        processor_fast.save_pretrained(self.tmpdirname)        

    def get_tokenizer(self, **kwargs):
        return CLIPTokenizer.from_pretrained(self.checkpoint, **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        return CLIPTokenizerFast.from_pretrained(self.checkpoint, **kwargs)

    def get_image_processor(self, **kwargs):
        return ImageBindImageProcessor.from_pretrained(self.checkpoint, **kwargs)

    def get_feature_extractor(self, **kwargs):
        return ImageBindFeatureExtractor.from_pretrained(self.checkpoint, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def prepare_image_inputs(self):
        """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
        or a list of PyTorch tensors if one specifies torchify=True.
        """

        image_inputs = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)]

        image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]

        return image_inputs

    def prepare_audio_inputs(self):
        return [np.random.rand(1500)]

    def test_save_load_pretrained_default(self):
        tokenizer_slow = self.get_tokenizer()
        tokenizer_fast = self.get_rust_tokenizer()
        image_processor = self.get_image_processor()
        feature_extractor = self.get_feature_extractor()

        processor_slow = ImageBindProcessor(
            tokenizer=tokenizer_slow, image_processor=image_processor, feature_extractor=feature_extractor
        )
        processor_slow.save_pretrained(self.tmpdirname)
        processor_slow = ImageBindProcessor.from_pretrained(self.tmpdirname, use_fast=False)

        processor_fast = ImageBindProcessor(
            tokenizer=tokenizer_fast, image_processor=image_processor, feature_extractor=feature_extractor
        )
        processor_fast.save_pretrained(self.tmpdirname)
        processor_fast = ImageBindProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor_slow.tokenizer.get_vocab(), tokenizer_slow.get_vocab())
        self.assertEqual(processor_fast.tokenizer.get_vocab(), tokenizer_fast.get_vocab())
        self.assertEqual(tokenizer_slow.get_vocab(), tokenizer_fast.get_vocab())
        self.assertIsInstance(processor_slow.tokenizer, CLIPTokenizer)
        self.assertIsInstance(processor_fast.tokenizer, CLIPTokenizerFast)

        self.assertEqual(processor_slow.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertEqual(processor_fast.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertIsInstance(processor_slow.image_processor, ImageBindImageProcessor)
        self.assertIsInstance(processor_fast.image_processor, ImageBindImageProcessor)

        self.assertEqual(processor_slow.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertEqual(processor_fast.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(processor_slow.feature_extractor, ImageBindFeatureExtractor)
        self.assertIsInstance(processor_fast.feature_extractor, ImageBindFeatureExtractor)

    def test_save_load_pretrained_additional_features(self):
        processor = ImageBindProcessor(
            tokenizer=self.get_tokenizer(),
            image_processor=self.get_image_processor(),
            feature_extractor=self.get_feature_extractor(),
        )
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        # Need to put same kwargs for both image_processor and feature_extractor as they share the same config :/
        image_processor_add_kwargs = self.get_image_processor(do_convert_rgb=False, do_chunk=False, num_chunks=5)
        feature_extractor_add_kwargs = self.get_feature_extractor(do_convert_rgb=False, do_chunk=False, num_chunks=5)

        processor = ImageBindProcessor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_convert_rgb=False, do_chunk=False, num_chunks=5
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, CLIPTokenizerFast)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, ImageBindImageProcessor)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.feature_extractor, ImageBindFeatureExtractor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = ImageBindProcessor(
            tokenizer=tokenizer, image_processor=image_processor, feature_extractor=self.get_feature_extractor()
        )

        image_input = self.prepare_image_inputs()

        input_image_proc = image_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, return_tensors="np")

        for key in input_image_proc.keys():
            self.assertAlmostEqual(input_image_proc[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_feature_extractor(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = ImageBindProcessor(
            tokenizer=tokenizer, feature_extractor=feature_extractor, image_processor=self.get_image_processor()
        )

        raw_speech = self.prepare_audio_inputs()

        input_feat_extract = feature_extractor(raw_speech, return_tensors="np")
        input_processor = processor(audio=raw_speech, return_tensors="np")

        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        processor = ImageBindProcessor(
            tokenizer=tokenizer, image_processor=image_processor, feature_extractor=feature_extractor
        )

        input_str = "lower newer"

        encoded_processor = processor(text=input_str)

        encoded_tok = tokenizer(input_str)

        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        processor = ImageBindProcessor(
            tokenizer=tokenizer, image_processor=image_processor, feature_extractor=feature_extractor
        )

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(list(inputs.keys()), ["input_ids", "attention_mask", "pixel_values"])

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    def test_tokenizer_decode(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        processor = ImageBindProcessor(
            tokenizer=tokenizer, image_processor=image_processor, feature_extractor=feature_extractor
        )

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        processor = ImageBindProcessor(
            tokenizer=tokenizer, image_processor=image_processor, feature_extractor=feature_extractor
        )

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        audio_input = self.prepare_audio_inputs()

        inputs = processor(text=input_str, images=image_input, audio=audio_input)

        self.assertListEqual(list(inputs.keys()), processor.model_input_names)

    @require_vision
    @require_torch
    def test_tokenizer_defaults_preserved_by_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer", max_length=117)
        feature_extractor = self.get_component("feature_extractor")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor, feature_extractor=feature_extractor)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, return_tensors="pt")
        self.assertEqual(len(inputs["input_ids"][0]), 4)

    @require_torch
    @require_vision
    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor", crop_size=(234, 234))
        tokenizer = self.get_component("tokenizer", max_length=117)
        feature_extractor = self.get_component("feature_extractor")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor, feature_extractor=feature_extractor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)
        self.assertEqual(len(inputs["pixel_values"][0][0]), 234)

    @require_vision
    @require_torch
    def test_kwargs_overrides_default_tokenizer_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer", max_length=117)
        feature_extractor = self.get_component("feature_extractor")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor, feature_extractor=feature_extractor)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, return_tensors="pt", max_length=112)
        self.assertEqual(len(inputs["input_ids"][0]), 4)

    @require_torch
    @require_vision
    def test_kwargs_overrides_default_image_processor_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor", crop_size=(234, 234))
        tokenizer = self.get_component("tokenizer", max_length=117)
        feature_extractor = self.get_component("feature_extractor")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor, feature_extractor=feature_extractor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, crop_size=[224, 224])
        self.assertEqual(len(inputs["pixel_values"][0][0]), 224)

    @require_torch
    @require_vision
    def test_unstructured_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")
        feature_extractor = self.get_component("feature_extractor")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor, feature_extractor=feature_extractor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            crop_size={"height": 214, "width": 214},
            padding="max_length",
            max_length=76,
        )

        self.assertEqual(inputs["pixel_values"].shape[2], 214)
        self.assertEqual(len(inputs["input_ids"][0]), 76)

    @require_torch
    @require_vision
    def test_unstructured_kwargs_batched(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")
        feature_extractor = self.get_component("feature_extractor")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor, feature_extractor=feature_extractor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = ["lower newer", "upper older longer string"]
        image_input = self.prepare_image_inputs() * 2
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            crop_size={"height": 214, "width": 214},
            padding="longest",
            max_length=76,
        )

        self.assertEqual(inputs["pixel_values"].shape[2], 214)

        self.assertEqual(len(inputs["input_ids"][0]), 6)

    @require_torch
    @require_vision
    def test_doubly_passed_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")
        feature_extractor = self.get_component("feature_extractor")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor, feature_extractor=feature_extractor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = ["lower newer"]
        image_input = self.prepare_image_inputs()
        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                images=image_input,
                images_kwargs={"crop_size": {"height": 222, "width": 222}},
                crop_size={"height": 214, "width": 214},
            )

    @require_torch
    @require_vision
    def test_structured_kwargs_nested(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")
        feature_extractor = self.get_component("feature_extractor")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor, feature_extractor=feature_extractor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"crop_size": {"height": 214, "width": 214}},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        self.assertEqual(inputs["pixel_values"].shape[2], 214)

        self.assertEqual(len(inputs["input_ids"][0]), 76)

    @require_torch
    @require_vision
    def test_structured_kwargs_nested_from_dict(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")

        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")
        feature_extractor = self.get_component("feature_extractor")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor, feature_extractor=feature_extractor)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"crop_size": {"height": 214, "width": 214}},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.assertEqual(inputs["pixel_values"].shape[2], 214)

        self.assertEqual(len(inputs["input_ids"][0]), 76)