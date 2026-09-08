from experiments.analyze_targets import _annotate_materialization


def record(day, selected, fertilize=True, position=(9, 0)):
    return {
        "day": day,
        "hour": 0,
        "position": list(position),
        "selected": selected,
        "fertilize": fertilize if selected else None,
    }


def planting(day, hour, name, position=(9, 0)):
    return {
        "day": day,
        "hour": hour,
        "position": list(position),
        "name": name,
        "planted_day": day,
        "placed_day": None,
    }


def test_materialization_stops_at_target_change():
    records = [
        record(8, "TOMATO"),
        record(9, "STRAWBERRY"),
        record(19, "TOMATO"),
        record(20, "TOMATO"),
    ]
    plantings = [
        planting(9, 16, "STRAWBERRY"),
        planting(20, 22, "TOMATO"),
    ]

    decision_counts, episode_counts, episodes = _annotate_materialization(
        records, plantings
    )

    assert records[0]["materialization"]["status"] == "superseded"
    assert records[0]["materialization"]["next_selected"] == "STRAWBERRY"
    assert records[1]["materialization"]["status"] == "same_day"
    assert records[2]["materialization"]["status"] == "continued"
    assert records[3]["materialization"]["status"] == "same_day"

    assert decision_counts == {
        "superseded": 1,
        "same_day": 2,
        "continued": 1,
    }
    assert episode_counts == {
        "not_materialized": 1,
        "same_day": 1,
        "delayed": 1,
    }

    first_tomato = episodes[0]
    assert first_tomato["selected"] == "TOMATO"
    assert first_tomato["status"] == "not_materialized"
    final_tomato = episodes[-1]
    assert final_tomato["selected"] == "TOMATO"
    assert final_tomato["status"] == "delayed"
    assert final_tomato["delay_days"] == 1
