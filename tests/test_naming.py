from modelbenchx import naming


def test_canonical_key_roundtrip():
    assert naming.canonical_key("sam2", "encoder") == "sam2__encoder"
    assert naming.split_key("sam2__encoder") == ("sam2", "encoder")


def test_split_key_component_with_double_underscore():
    # The first separator wins; the component keeps any further "__".
    assert naming.split_key("a__b__c") == ("a", "b__c")


def test_model_from_onnx_zip():
    assert naming.model_from_onnx_zip("squeezenet1_1-onnx-float.zip") == "squeezenet1_1"
    assert naming.model_from_onnx_zip("sam2-onnx-float") == "sam2"


def test_onnx_member_key_single_and_multi():
    # Single-graph zip: the onnx file is named after the model.
    assert naming.onnx_member_key("squeezenet1_1", "squeezenet1_1.onnx") == "squeezenet1_1__squeezenet1_1"
    # Multi-graph zip: component is the onnx stem.
    assert naming.onnx_member_key("sam2", "encoder.onnx") == "sam2__encoder"
    # Path inside the archive is tolerated.
    assert naming.onnx_member_key("sam2", "sam2-onnx-float/decoder.onnx") == "sam2__decoder"


def test_key_from_coreml_filename():
    assert naming.key_from_coreml_filename("sam2__encoder.mlpackage") == "sam2__encoder"
    assert naming.key_from_coreml_filename("squeezenet1_1__squeezenet1_1.mlmodel") == "squeezenet1_1__squeezenet1_1"


def test_key_from_aimodel_dirname():
    assert naming.key_from_aimodel_dirname("sam2-onnx-float__encoder.aimodel") == "sam2__encoder"
    assert (
        naming.key_from_aimodel_dirname("squeezenet1_1-onnx-float__squeezenet1_1.aimodel")
        == "squeezenet1_1__squeezenet1_1"
    )
    # deepbox keeps a component that differs from the model name.
    assert (
        naming.key_from_aimodel_dirname("deepbox-onnx-float__vgg_3d_detection.aimodel")
        == "deepbox__vgg_3d_detection"
    )


def test_all_four_formats_agree_on_key():
    keys = {
        naming.onnx_member_key("sam2", "encoder.onnx"),
        naming.key_from_coreml_filename("sam2__encoder.mlpackage"),
        naming.key_from_coreml_filename("sam2__encoder.mlmodel"),
        naming.key_from_aimodel_dirname("sam2-onnx-float__encoder.aimodel"),
    }
    assert keys == {"sam2__encoder"}


def test_sanitize_feature_name():
    assert naming.sanitize_feature_name("/head/Add_output_0") == "_head_Add_output_0"
    assert naming.sanitize_feature_name("class.logits") == "class_logits"
    assert naming.sanitize_feature_name("123abc") == "_123abc"


def test_match_output_names_exact_sanitized_positional():
    # exact
    assert naming.match_output_names(["a", "b"], ["a", "b"]) == {"a": "a", "b": "b"}
    # sanitized
    assert naming.match_output_names(["x.y"], ["x_y"]) == {"x.y": "x_y"}
    # positional fallback when counts match and names are opaque
    assert naming.match_output_names(["out0", "out1"], ["var_42", "var_99"]) == {
        "out0": "var_42",
        "out1": "var_99",
    }


def test_match_output_names_positional_fallback_skips_consumed_actuals():
    # 'b' matches exactly (at a different index); leftover ref 'a' must take the
    # leftover actual 'c', not re-grab the already-consumed 'b'.
    assert naming.match_output_names(["a", "b"], ["b", "c"]) == {"a": "c", "b": "b"}
