async def test_health_endpoint(client):
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "queue_lag" in body
    assert "db" in body
