import numpy as np

from robot.perception.contact_detection_adapter import (
    associate_contact_parts_to_objects,
)
from robot.perception.contact_replay import evaluate_contact_replay
from robot.perception.semantic_detection import (
    PixelBoundingBox,
    SemanticDetection,
    SemanticDetectionFrame,
)


def detection(name, mask):
    ys, xs = np.nonzero(mask)
    return SemanticDetection(
        0,
        name,
        0.8,
        PixelBoundingBox(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1),
        mask,
    )


def frame(*items, model="model.pt"):
    return SemanticDetectionFrame(0, 0, 12, 12, items, model, 0.01)


def test_positive_replay_reports_accepted_parent_without_expanding_scope():
    parent = np.zeros((12, 12), dtype=bool)
    parent[1:11, 1:11] = True
    edge = np.zeros((12, 12), dtype=bool)
    edge[2:4, 2:10] = True
    objects = frame(detection("knife", parent), model="objects.pt")
    parts = frame(detection("exposed_edge", edge), model="contact.pt")
    association = associate_contact_parts_to_objects(objects, parts)

    report = evaluate_contact_replay(
        objects,
        parts,
        association,
        image_sha256="a" * 64,
        object_checkpoint_sha256="b" * 64,
        contact_checkpoint_sha256="c" * 64,
    )
    assert report.positive_association_valid
    assert report.accepted_parent_classes == ("knife",)
    assert any("not physical sharpness" in item for item in report.limitations)


def test_empty_replay_is_not_a_positive_association():
    objects = frame(model="objects.pt")
    parts = frame(model="contact.pt")
    association = associate_contact_parts_to_objects(objects, parts)

    report = evaluate_contact_replay(
        objects,
        parts,
        association,
        image_sha256="a" * 64,
        object_checkpoint_sha256="b" * 64,
        contact_checkpoint_sha256="c" * 64,
    )
    assert not report.positive_association_valid
    assert report.accepted_parent_classes == ()
