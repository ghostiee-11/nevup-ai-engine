import pytest

from scripts.seed import seed


@pytest.mark.integration
async def test_seed_loads_all_records(db_clean):
    counts = await seed()
    assert counts["traders"] == 10
    assert counts["sessions"] == 52
    assert counts["trades"] == 388
