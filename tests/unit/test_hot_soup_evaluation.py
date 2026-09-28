from robot.perception.hot_soup_dataset import HotSoupBox, HotSoupSample
from robot.perception.hot_soup_evaluation import (
    HotSoupPrediction,
    evaluate_hot_soup_predictions,
    predictions_from_hot_soup_frame,
)
from robot.perception.hot_soup_detector import HotSoupDetection, HotSoupDetectionFrame
from robot.perception.semantic_detection import PixelBoundingBox


def box(name, x0, y0, x1, y1):
    return HotSoupBox(
        name,
        x0,
        y0,
        x1,
        y1,
        "human_visual_annotation",
    )


def sample(identifier, boxes=(), hard=()):
    present = any(item.class_name == "visible_hot_soup_plume" for item in boxes)
    container = any(item.class_name == "open_container" for item in boxes)
    return HotSoupSample(
        identifier,
        f"group-{identifier}",
        "test",
        f"{identifier}.png",
        "a" * 64,
        100,
        100,
        "real_user_owned",
        "owned test capture",
        "owned by project",
        "present" if present else "absent",
        "present" if container else "absent",
        boxes,
        hard,
    )


def prediction(name, confidence=0.9, coordinates=(0.2, 0.1, 0.5, 0.4)):
    return HotSoupPrediction(name, confidence, *coordinates)


def metric(report, name):
    return next(item for item in report.class_metrics if item.class_name == name)


def test_per_class_metrics_and_hard_negative_errors_remain_separate():
    positive = sample(
        "positive",
        (
            box("visible_hot_soup_plume", 0.2, 0.1, 0.5, 0.4),
            box("open_container", 0.2, 0.4, 0.6, 0.8),
        ),
    )
    negative = sample("glare", hard=("glare",))
    report = evaluate_hot_soup_predictions(
        (positive, negative),
        {
            "positive": (
                prediction("visible_hot_soup_plume"),
                prediction("open_container", coordinates=(0.2, 0.4, 0.6, 0.8)),
            ),
            "glare": (prediction("visible_hot_soup_plume"),),
        },
    )
    plume = metric(report, "visible_hot_soup_plume")
    container = metric(report, "open_container")
    assert (plume.true_positive_count, plume.false_positive_count) == (1, 1)
    assert plume.precision == 0.5 and plume.recall == 1.0
    assert container.f1 == 1.0
    assert report.hard_negative_plume_false_positive_count == 1


def test_low_confidence_and_missing_predictions_become_false_negatives():
    positive = sample(
        "positive",
        (
            box("visible_hot_soup_plume", 0.2, 0.1, 0.5, 0.4),
            box("open_container", 0.2, 0.4, 0.6, 0.8),
        ),
    )
    report = evaluate_hot_soup_predictions(
        (positive,),
        {"positive": (prediction("visible_hot_soup_plume", confidence=0.2),)},
    )
    assert metric(report, "visible_hot_soup_plume").false_negative_count == 1
    assert metric(report, "open_container").false_negative_count == 1


def test_runtime_detection_frame_converts_to_normalized_evaluation_boxes():
    frame = HotSoupDetectionFrame(
        1,
        2.0,
        100,
        200,
        (
            HotSoupDetection(
                0,
                "visible_hot_soup_plume",
                0.8,
                PixelBoundingBox(20, 10, 100, 40),
            ),
        ),
        "fixture.pt",
        0.01,
    )
    result = predictions_from_hot_soup_frame(frame)
    assert result[0].x_min == 0.1
    assert result[0].y_max == 0.4
