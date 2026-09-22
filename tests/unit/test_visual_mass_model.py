import math

from PIL import Image
import pytest
import torch

from robot.perception.abo_visual_mass import AboMassCandidate
from robot.perception.visual_mass_model import (
    VISUAL_MASS_ARCHITECTURE,
    VISUAL_MASS_MODEL_SCHEMA_VERSION,
    AboVisualMassDataset,
    VisualMassRegressor,
    fit_log_dimension_baseline,
    log_dimension_statistics,
    load_visual_mass_checkpoint,
    mass_regression_metrics,
    predict_log_dimension_baseline,
)


def candidate(**changes):
    values = dict(
        object_id="amazon.com:a",
        split_group_id="a",
        item_id="a",
        domain_name="amazon.com",
        semantic_class="chair",
        product_type="CHAIR",
        image_id="image-a",
        image_path="aa/a.jpg",
        mass_kg=4,
        width_m=0.5,
        depth_m=0.6,
        height_m=0.8,
        split="train",
    )
    values.update(changes)
    return AboMassCandidate(**values)


def test_log_dimension_statistics_use_positive_si_dimensions():
    mean, std = log_dimension_statistics(
        [candidate(), candidate(object_id="b", width_m=1, depth_m=1.2, height_m=1.6)]
    )
    assert mean == pytest.approx(tuple(math.log(value) for value in (2**-0.5, 0.72**0.5, 1.28**0.5)))
    assert all(value > 0 for value in std)


def test_model_fuses_image_and_dimension_features_and_predicts_positive_mass():
    model = VisualMassRegressor((0, 0, 0), (1, 1, 1), pretrained=False)
    model.eval()
    images = torch.zeros((2, 3, 224, 224))
    dimensions = torch.tensor(((0.5, 0.6, 0.8), (1.0, 1.0, 1.0)))
    with torch.no_grad():
        log_mass = model(images, dimensions)
        mass = model.predict_mass_kg(images, dimensions)
    assert log_mass.shape == (2,)
    assert mass.shape == (2,)
    assert torch.all(torch.isfinite(mass)) and torch.all(mass > 0)


def test_frozen_backbone_keeps_batch_norm_in_evaluation_mode():
    model = VisualMassRegressor(
        (0, 0, 0), (1, 1, 1), pretrained=False, freeze_backbone=True
    )
    model.train()
    assert model.regressor.training
    assert not model.backbone.training


def test_model_rejects_nonpositive_or_malformed_dimensions():
    model = VisualMassRegressor((0, 0, 0), (1, 1, 1), pretrained=False)
    images = torch.zeros((1, 3, 224, 224))
    with pytest.raises(ValueError):
        model(images, torch.tensor(((1.0, 0.0, 1.0),)))
    with pytest.raises(ValueError):
        model(images, torch.ones((1, 2)))


def test_checkpoint_loader_validates_contract_without_downloading_weights(tmp_path):
    model = VisualMassRegressor(
        (0, 0, 0), (1, 1, 1), pretrained=False, freeze_backbone=True
    )
    path = tmp_path / "model.pt"
    torch.save(
        {
            "schema_version": VISUAL_MASS_MODEL_SCHEMA_VERSION,
            "architecture": VISUAL_MASS_ARCHITECTURE,
            "state_dict": model.state_dict(),
            "dimension_mean": (0, 0, 0),
            "dimension_std": (1, 1, 1),
            "pretrained": False,
            "backbone_frozen": True,
        },
        path,
    )
    loaded, metadata = load_visual_mass_checkpoint(path)
    assert not loaded.training
    assert not loaded.backbone.training
    assert metadata["architecture"] == VISUAL_MASS_ARCHITECTURE

    torch.save({"schema_version": 999}, path)
    with pytest.raises(ValueError, match="schema"):
        load_visual_mass_checkpoint(path)


def test_dataset_loads_image_and_keeps_external_label_offline(tmp_path):
    item = candidate()
    path = tmp_path / item.image_path
    path.parent.mkdir(parents=True)
    Image.new("RGB", (256, 256), color=(10, 20, 30)).save(path)
    dataset = AboVisualMassDataset([item], tmp_path, training=False)
    pixels, dimensions, log_mass, object_id, semantic_class = dataset[0]
    assert pixels.shape == (3, 224, 224)
    assert dimensions.tolist() == pytest.approx([0.5, 0.6, 0.8])
    assert float(log_mass) == pytest.approx(math.log(4))
    assert object_id == "amazon.com:a"
    assert semantic_class == "chair"
    assert item.runtime_evidence_status == "unavailable"


def test_metrics_are_reported_in_physical_mass_units():
    metrics = mass_regression_metrics(
        torch.log(torch.tensor((1.0, 4.0))),
        torch.log(torch.tensor((2.0, 2.0))),
    )
    assert metrics.mean_absolute_error_kg == pytest.approx(1.5)
    assert metrics.median_absolute_error_kg == pytest.approx(1.5)
    assert metrics.mean_absolute_percentage_error == pytest.approx(0.75)


def test_dimension_only_baseline_recovers_log_linear_relation():
    items = []
    for index, dimensions in enumerate(
        ((0.2, 0.3, 0.4), (0.3, 0.4, 0.5), (0.5, 0.7, 0.9), (1.0, 0.8, 0.6), (1.2, 1.3, 1.4))
    ):
        mass = 2.0 * dimensions[0] * dimensions[1] * dimensions[2]
        items.append(
            candidate(
                object_id=str(index),
                item_id=str(index),
                split_group_id=str(index),
                mass_kg=mass,
                width_m=dimensions[0],
                depth_m=dimensions[1],
                height_m=dimensions[2],
            )
        )
    coefficients = fit_log_dimension_baseline(items)
    prediction = predict_log_dimension_baseline(items, coefficients)
    target = torch.tensor([math.log(item.mass_kg) for item in items])
    assert coefficients.tolist() == pytest.approx([math.log(2), 1, 1, 1], abs=1e-5)
    assert prediction.tolist() == pytest.approx(target.tolist(), abs=1e-5)
