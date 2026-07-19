from decimal import Decimal

import pytest
from dataclasses import FrozenInstanceError

from test_ocr_foundation import make_graph

from orderflowgpt_genesis import (
    BoundingBox,
    CellSemanticRole,
    DeterministicNumericParser,
    DeterministicOCRPostProcessor,
    NumericType,
    OCRMetadata,
    OCRNormalizationConfiguration,
    OCRResult,
    ParsedValue,
    ParsingError,
    ParsingResult,
    errors,
    is_empty,
    is_numeric,
    is_valid,
    normalize,
    numeric_value,
    parse,
    warnings,
)


def ocr(text: str, cell_id: str = "cell-1", confidence: float = 0.9) -> OCRResult:
    return OCRResult(
        "frame-1",
        cell_id,
        CellSemanticRole.ASK_REGION,
        text,
        confidence,
        (BoundingBox(1, 1, 10, 10),),
        metadata=OCRMetadata("engine", "provider"),
    )


def test_configuration_and_models_are_immutable():
    config = OCRNormalizationConfiguration(replacement_table={"x": "1"})
    with pytest.raises(TypeError):
        config.replacement_table["y"] = "2"
    value = DeterministicOCRPostProcessor(config).process(ocr("x"))
    assert value.numeric_value() == 1
    with pytest.raises(FrozenInstanceError):
        value.parsed_value.raw_text = "2"


def test_normalization_character_replacement_whitespace_unicode_and_separators():
    result = DeterministicOCRPostProcessor().process(ocr(" O l S B 1,234−"))
    assert result.parsed_value.normalized_text == "01581234-"
    assert not result.success
    assert "whitespace removed" in result.warnings
    assert "garbage trimmed" not in result.warnings
    assert normalize("−1٫25") == "-1.25"


def test_parse_integers_decimals_and_signed_values():
    cases = {
        "123": (123, NumericType.INTEGER),
        "123.45": (Decimal("123.45"), NumericType.DECIMAL),
        "-123": (-123, NumericType.SIGNED_INTEGER),
        "-123.45": (Decimal("-123.45"), NumericType.SIGNED_DECIMAL),
    }
    for text, (expected, numeric_type) in cases.items():
        result = parse(text)
        assert result.success
        assert result.parsed_value.numeric_type is numeric_type
        assert result.numeric_value() == expected
        assert is_numeric(result)
        assert is_valid(result)


def test_invalid_malformed_empty_and_restricted_values():
    invalid = ["", "abc", "1-2", "--1", "1.2.3", ".5", "5.", "NaN", "Infinity"]
    for text in invalid:
        result = DeterministicNumericParser().parse(text)
        assert not result.success
        assert result.parsed_value.numeric_type in {
            NumericType.INVALID,
            NumericType.EMPTY,
        }
    assert is_empty(parse(""))
    assert (
        not DeterministicNumericParser(
            OCRNormalizationConfiguration(allow_negative=False)
        )
        .parse("-1")
        .success
    )
    assert (
        not DeterministicNumericParser(
            OCRNormalizationConfiguration(allow_decimal=False)
        )
        .parse("1.1")
        .success
    )


def test_length_overflow_warning_generation_and_helper_accessors():
    config = OCRNormalizationConfiguration(maximum_length=3)
    result = parse("1234", config)
    assert not result.success
    assert errors(result)[0].reason == "overflow"
    processed = DeterministicOCRPostProcessor().process(ocr("abc123xyz"))
    assert processed.parsed_value.normalized_text == "123"
    assert "garbage trimmed" in warnings(processed)
    assert numeric_value(processed) == 123


def test_duplicate_sign_and_multiple_decimal_normalization_pipeline_ordering():
    processor = DeterministicOCRPostProcessor()
    result = processor.process(ocr("--1.2.3"))
    assert result.success
    assert result.parsed_value.normalized_text == "-1.23"
    assert result.numeric_value() == Decimal("-1.23")
    assert "multiple decimal separators normalized" in result.warnings
    assert "duplicate sign cleanup" in result.warnings


def test_graph_parsed_value_lookup_helpers():
    graph = make_graph(1, 1)
    assert graph.parsed_values
    cell_id = graph.ocr_results[0].cell_id
    assert graph.lookup_parsed(cell_id)
    assert graph.lookup_numeric(cell_id) == ()
    assert graph.lookup_invalid() == graph.parsed_values


def test_parsing_result_validation_requires_failure_reason():
    parsed = ParsedValue("", "", NumericType.EMPTY, None, 0.0)
    with pytest.raises(ValueError, match="failure reason"):
        ParsingResult(parsed, False)
    assert ParsingError("bad", "x", 0).position == 0
