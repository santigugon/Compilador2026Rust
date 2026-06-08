from __future__ import annotations

import sys
import types
import unittest

from pipeline.serve.vllm_engine import _add_guided_json_sampling_param


class _StructuredOutputsParams:
    def __init__(self, *, json: dict) -> None:
        self.json = json


class _GuidedDecodingParams:
    def __init__(self, *, json: dict) -> None:
        self.json = json


class _GuidedDecodingParamsJsonSchemaOnly:
    def __init__(self, *, json_schema: dict) -> None:
        self.json_schema = json_schema


class _SamplingParamsStructured:
    def __init__(self, *, structured_outputs: object | None = None) -> None:
        pass


class _SamplingParamsGuidedDecoding:
    def __init__(self, *, guided_decoding: object | None = None) -> None:
        pass


class _SamplingParamsGuidedJson:
    def __init__(self, *, guided_json: dict | None = None) -> None:
        pass


class _SamplingParamsUnsupported:
    def __init__(self, *, max_tokens: int | None = None) -> None:
        pass


class GuidedJsonSamplingParamTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old_vllm = sys.modules.get("vllm")
        self._old_sampling_params = sys.modules.get("vllm.sampling_params")

    def tearDown(self) -> None:
        if self._old_vllm is None:
            sys.modules.pop("vllm", None)
        else:
            sys.modules["vllm"] = self._old_vllm

        if self._old_sampling_params is None:
            sys.modules.pop("vllm.sampling_params", None)
        else:
            sys.modules["vllm.sampling_params"] = self._old_sampling_params

    def _install_fake_vllm(
        self,
        *,
        structured_cls: type = _StructuredOutputsParams,
        guided_cls: type = _GuidedDecodingParams,
    ) -> None:
        vllm_module = types.ModuleType("vllm")
        sampling_module = types.ModuleType("vllm.sampling_params")
        sampling_module.StructuredOutputsParams = structured_cls
        sampling_module.GuidedDecodingParams = guided_cls
        sys.modules["vllm"] = vllm_module
        sys.modules["vllm.sampling_params"] = sampling_module

    def test_uses_structured_outputs_when_available(self) -> None:
        self._install_fake_vllm()
        schema = {"type": "object"}
        sampling_kwargs: dict = {}

        _add_guided_json_sampling_param(
            sampling_kwargs,
            _SamplingParamsStructured,
            schema,
        )

        structured = sampling_kwargs["structured_outputs"]
        self.assertIsInstance(structured, _StructuredOutputsParams)
        self.assertEqual(structured.json, schema)

    def test_uses_guided_decoding_when_available(self) -> None:
        self._install_fake_vllm()
        schema = {"type": "object"}
        sampling_kwargs: dict = {}

        _add_guided_json_sampling_param(
            sampling_kwargs,
            _SamplingParamsGuidedDecoding,
            schema,
        )

        guided = sampling_kwargs["guided_decoding"]
        self.assertIsInstance(guided, _GuidedDecodingParams)
        self.assertEqual(guided.json, schema)

    def test_uses_guided_decoding_json_schema_fallback(self) -> None:
        self._install_fake_vllm(guided_cls=_GuidedDecodingParamsJsonSchemaOnly)
        schema = {"type": "object"}
        sampling_kwargs: dict = {}

        _add_guided_json_sampling_param(
            sampling_kwargs,
            _SamplingParamsGuidedDecoding,
            schema,
        )

        guided = sampling_kwargs["guided_decoding"]
        self.assertIsInstance(guided, _GuidedDecodingParamsJsonSchemaOnly)
        self.assertEqual(guided.json_schema, schema)

    def test_uses_guided_json_when_available(self) -> None:
        schema = {"type": "object"}
        sampling_kwargs: dict = {}

        _add_guided_json_sampling_param(
            sampling_kwargs,
            _SamplingParamsGuidedJson,
            schema,
        )

        self.assertEqual(sampling_kwargs["guided_json"], schema)

    def test_raises_when_no_guided_api_is_available(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "structured_outputs"):
            _add_guided_json_sampling_param(
                {},
                _SamplingParamsUnsupported,
                {"type": "object"},
            )


if __name__ == "__main__":
    unittest.main()
